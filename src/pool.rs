use crate::Failure;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashSet,
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
    policy: Policy,
    client: reqwest::Client,
    changed: Notify,
    per_worker: usize,
}
#[derive(Deserialize)]
pub struct WorkerOutput {
    pub output: Vec<f32>,
}
pub struct ResultOutput {
    pub worker_id: usize,
    pub output: Vec<f32>,
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
    fn select(self: &Arc<Self>, excluded: &HashSet<usize>) -> Option<Lease> {
        let mut state = self.state.lock().unwrap();
        let n = state.workers.len();
        let mut eligible: Vec<_> = (0..n)
            .map(|offset| (state.cursor + offset) % n)
            .filter(|&i| {
                let w = &state.workers[i];
                w.healthy && w.in_flight < self.per_worker && !excluded.contains(&i)
            })
            .collect();
        if self.policy == Policy::LeastInFlight {
            eligible.sort_by_key(|&i| state.workers[i].in_flight);
        }
        let id = *eligible.first()?;
        state.cursor = (id + 1) % n;
        state.workers[id].in_flight += 1;
        Some(Lease {
            pool: self.clone(),
            id,
        })
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
        let mut excluded = HashSet::new();
        // At most two attempts, only on different workers. The operation is pure;
        // a lost response can cause duplicate computation but no side effects.
        for _ in 0..2 {
            let lease = loop {
                if Instant::now() >= deadline {
                    return Err(Failure::Deadline);
                }
                let notified = self.changed.notified();
                tokio::pin!(notified);
                notified.as_mut().enable(); // Register before checking to avoid lost wakeups.
                if let Some(lease) = self.select(&excluded) {
                    break lease;
                }
                if !self
                    .snapshot()
                    .iter()
                    .any(|w| w.healthy && !excluded.contains(&w.id))
                {
                    return Err(Failure::Unavailable);
                }
                timeout_at(deadline, notified)
                    .await
                    .map_err(|_| Failure::Deadline)?;
            };
            let id = lease.id;
            excluded.insert(id);
            let url = self.snapshot()[id].url.clone();
            // Give a stalled first attempt only half the remaining budget, leaving
            // room for one retry. Every stage remains inside the original deadline.
            let remaining = deadline.saturating_duration_since(Instant::now());
            let attempt_budget = if excluded.len() == 1 {
                remaining / 2
            } else {
                remaining
            };
            let result = async {
                let response = self.client.post(format!("{url}/infer"))
                    .json(&serde_json::json!({"values":values,"request_id":request_id,"remaining_ms":attempt_budget.as_millis().clamp(1,10000) as u64}))
                    .send().await?.error_for_status()?;
                response.json::<WorkerOutput>().await
            };
            if let Ok(Ok(body)) = tokio::time::timeout(attempt_budget, result).await {
                if body.output.len() == 2 && body.output.iter().all(|v| v.is_finite()) {
                    return Ok(ResultOutput {
                        worker_id: id,
                        output: body.output,
                    });
                }
            }
            self.mark_failed(id);
            drop(lease);
        }
        Err(if Instant::now() >= deadline {
            Failure::Deadline
        } else {
            Failure::Unavailable
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
        let a = p.select(&HashSet::new()).unwrap();
        let b = p.select(&HashSet::new()).unwrap();
        assert_eq!((a.id, b.id), (0, 1));
        p.mark_failed(0);
        assert_eq!(p.select(&HashSet::new()).unwrap().id, 1);
        drop(a);
        drop(b);
        assert!(p.snapshot().iter().all(|w| w.in_flight == 0));
    }
    #[test]
    fn least_in_flight_and_exclusion() {
        let p = healthy(Policy::LeastInFlight);
        let a = p.select(&HashSet::new()).unwrap();
        p.state.lock().unwrap().cursor = 0;
        assert_eq!(p.select(&HashSet::new()).unwrap().id, 1);
        assert_eq!(p.select(&HashSet::from([0])).unwrap().id, 1);
        drop(a);
    }
}
