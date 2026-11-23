use crate::Failure;
use serde::{Deserialize, Serialize};
use std::{
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio::{
    sync::Notify,
    time::{timeout_at, Instant},
};

#[derive(Clone)]
pub enum Backend {
    /// This is an async sleep, never ONNX inference.
    Simulated {
        delay: Duration,
        fail: bool,
    },
    Workers(Arc<Pool>),
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Policy {
    RoundRobin,
    LeastInFlight,
}
impl std::str::FromStr for Policy {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "round-robin" => Ok(Self::RoundRobin),
            "least-in-flight" => Ok(Self::LeastInFlight),
            _ => Err("policy must be round-robin or least-in-flight".into()),
        }
    }
}
#[derive(Serialize, Clone)]
pub struct WorkerStatus {
    pub id: usize,
    pub url: String,
    pub healthy: bool,
    pub in_flight: usize,
}
struct State {
    workers: Vec<WorkerStatus>,
    cursor: usize,
    epochs: Vec<u64>,
}
pub struct Pool {
    state: Mutex<State>,
    infer_urls: Vec<String>,
    policy: Policy,
    client: reqwest::Client,
    changed: Notify,
    per_worker: usize,
}
#[derive(Deserialize)]
pub struct WorkerOutput {
    pub output: [f32; 2],
}
pub struct ResultOutput {
    pub worker_queue_us: Option<u64>,
    pub ort_us: Option<u64>,
    pub worker_id: usize,
    pub output: [f32; 2],
}
enum WorkerReply {
    Prediction(ResultOutput),
    Rejected(Failure),
}
#[derive(Serialize)]
struct WorkerInput<'a> {
    values: &'a [f32],
    request_id: &'a str,
    remaining_ms: u64,
}
// The guard owns its Arc, so cancellation drops it and decrements the count.
struct Lease {
    pool: Arc<Pool>,
    id: usize,
}
impl Drop for Lease {
    fn drop(&mut self) {
        self.pool.state.lock().unwrap().workers[self.id].in_flight -= 1;
        self.pool.changed.notify_waiters();
    }
}
impl Pool {
    pub fn new(urls: Vec<String>, policy: Policy, per_worker: usize) -> Arc<Self> {
        assert!(!urls.is_empty() && per_worker > 0);
        let n = urls.len();
        Arc::new(Self {
            infer_urls: urls.iter().map(|url| format!("{url}/infer")).collect(),
            state: Mutex::new(State {
                workers: urls
                    .into_iter()
                    .enumerate()
                    .map(|(id, url)| WorkerStatus {
                        id,
                        url,
                        healthy: false,
                        in_flight: 0,
                    })
                    .collect(),
                cursor: 0,
                epochs: vec![0; n],
            }),
            policy,
            client: reqwest::Client::builder()
                .no_proxy()
                .connect_timeout(Duration::from_millis(100))
                .build()
                .unwrap(),
            changed: Notify::new(),
            per_worker,
        })
    }
    pub fn snapshot(&self) -> Vec<WorkerStatus> {
        self.state.lock().unwrap().workers.clone()
    }
    // Ok(None) means healthy workers exist but all are at capacity.
    fn select(self: &Arc<Self>, excluded: Option<usize>) -> Result<Option<Lease>, Failure> {
        let mut state = self.state.lock().unwrap();
        let n = state.workers.len();
        let mut has_healthy_worker = false;
        let mut best = None;
        let mut best_load = self.per_worker;
        for offset in 0..n {
            let id = (state.cursor + offset) % n;
            let worker = &state.workers[id];
            if !worker.healthy || Some(id) == excluded {
                continue;
            }
            has_healthy_worker = true;
            if worker.in_flight < best_load {
                best = Some(id);
                best_load = worker.in_flight;
                if self.policy == Policy::RoundRobin || best_load == 0 {
                    break;
                }
            }
        }
        let Some(id) = best else {
            return if has_healthy_worker {
                Ok(None)
            } else {
                Err(Failure::Unavailable)
            };
        };
        state.cursor = (id + 1) % n;
        state.workers[id].in_flight += 1;
        Ok(Some(Lease {
            pool: self.clone(),
            id,
        }))
    }
    fn mark_failed(&self, id: usize) {
        let mut state = self.state.lock().unwrap();
        state.workers[id].healthy = false;
        state.epochs[id] += 1;
        self.changed.notify_waiters();
    }
    pub async fn probe(&self) {
        // Concurrent probes: one dead worker must not serialize all recovery.
        let snapshots = {
            let state = self.state.lock().unwrap();
            state
                .workers
                .iter()
                .map(|w| (w.id, w.url.clone(), state.epochs[w.id]))
                .collect::<Vec<_>>()
        };
        let mut probes = tokio::task::JoinSet::new();
        for (id, url, epoch) in snapshots {
            let client = self.client.clone();
            probes.spawn(async move {
                let healthy = client
                    .get(format!("{url}/health"))
                    .timeout(Duration::from_millis(150))
                    .send()
                    .await
                    .is_ok_and(|r| r.status().is_success());
                (id, epoch, healthy)
            });
        }
        while let Some(Ok((id, epoch, healthy))) = probes.join_next().await {
            let mut state = self.state.lock().unwrap();
            // Do not let a stale successful probe undo a newer inference failure.
            if state.epochs[id] == epoch {
                state.workers[id].healthy = healthy;
            }
        }
        self.changed.notify_waiters();
    }
    pub async fn infer(
        self: &Arc<Self>,
        values: &[f32],
        request_id: &str,
        deadline: Instant,
    ) -> Result<ResultOutput, Failure> {
        let mut excluded = None;
        let mut last_failure = Failure::Unavailable;
        // At most two attempts, only on different workers. The operation is pure;
        // a lost response can cause duplicate computation but no side effects.
        for _ in 0..2 {
            let lease = loop {
                if Instant::now() >= deadline {
                    return Err(Failure::Deadline);
                }
                // notify_waiters wakes even an unpolled Notified created before
                // the notification. No waiter registration on the ready path.
                let notified = self.changed.notified();
                match self.select(excluded) {
                    Ok(Some(lease)) => break lease,
                    Ok(None) => {}
                    Err(_) => return Err(last_failure),
                }
                timeout_at(deadline, notified)
                    .await
                    .map_err(|_| Failure::Deadline)?;
            };
            let id = lease.id;
            // Give a stalled first attempt only half the remaining budget, leaving
            // room for one retry. Every stage remains inside the original deadline.
            let remaining = deadline.saturating_duration_since(Instant::now());
            let attempt_budget = if excluded.is_none() && self.infer_urls.len() > 1 {
                remaining / 2
            } else {
                remaining
            };
            let result = async {
                let response = self
                    .client
                    .post(&self.infer_urls[id])
                    .json(&WorkerInput {
                        values,
                        request_id,
                        remaining_ms: attempt_budget.as_millis().clamp(1, 10000) as u64,
                    })
                    .send()
                    .await?;
                match response.status() {
                    reqwest::StatusCode::TOO_MANY_REQUESTS => {
                        return Ok(WorkerReply::Rejected(Failure::Overloaded));
                    }
                    reqwest::StatusCode::GATEWAY_TIMEOUT => {
                        return Ok(WorkerReply::Rejected(Failure::Deadline));
                    }
                    _ => {}
                }
                let response = response.error_for_status()?;
                let timing = |name| {
                    response
                        .headers()
                        .get(name)
                        .and_then(|v| v.to_str().ok()?.parse::<u64>().ok())
                };
                let worker_queue_us = timing("x-worker-queue-us");
                let ort_us = timing("x-onnx-run-us");
                response.json::<WorkerOutput>().await.map(|body| {
                    WorkerReply::Prediction(ResultOutput {
                        worker_id: id,
                        output: body.output,
                        worker_queue_us,
                        ort_us,
                    })
                })
            };
            match tokio::time::timeout(attempt_budget, result).await {
                Ok(Ok(WorkerReply::Prediction(result)))
                    if result.output.iter().all(|v| v.is_finite()) =>
                {
                    return Ok(result);
                }
                // A full worker or a job that expired in its executor queue
                // can leave the process healthy for the next request.
                Ok(Ok(WorkerReply::Rejected(reason))) => last_failure = reason,
                _ => {
                    self.mark_failed(id);
                    last_failure = Failure::Unavailable;
                }
            }
            excluded = Some(id);
            drop(lease);
        }
        Err(if Instant::now() >= deadline {
            Failure::Deadline
        } else {
            last_failure
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn healthy(policy: Policy) -> Arc<Pool> {
        let p = Pool::new(vec!["http://a".into(), "http://b".into()], policy, 2);
        for w in &mut p.state.lock().unwrap().workers {
            w.healthy = true;
        }
        p
    }
    #[test]
    fn round_robin_excludes_failed_and_releases_leases() {
        let p = healthy(Policy::RoundRobin);
        let a = p.select(None).unwrap().unwrap();
        let b = p.select(None).unwrap().unwrap();
        assert_eq!((a.id, b.id), (0, 1));
        p.mark_failed(0);
        assert_eq!(p.select(None).unwrap().unwrap().id, 1);
        drop(a);
        drop(b);
        assert!(p.snapshot().iter().all(|w| w.in_flight == 0));
    }
    #[test]
    fn least_in_flight_and_exclusion() {
        let p = healthy(Policy::LeastInFlight);
        let a = p.select(None).unwrap().unwrap();
        p.state.lock().unwrap().cursor = 0;
        assert_eq!(p.select(None).unwrap().unwrap().id, 1);
        assert_eq!(p.select(Some(0)).unwrap().unwrap().id, 1);
        drop(a);
    }
}
