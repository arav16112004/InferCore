//! InferCore: bounded admission, deadline propagation, and worker selection.
pub mod pool;
use axum::{
    extract::{DefaultBodyLimit, State},
    http::{HeaderName, HeaderValue, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use pool::{Backend, Pool};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{sync::Arc, time::Duration};
use tokio::{
    sync::{OwnedSemaphorePermit, Semaphore},
    time::{timeout_at, Instant},
};
use uuid::Uuid;

#[derive(Clone)]
pub struct AppState {
    active: Arc<Semaphore>,
    waiting: Arc<Semaphore>,
    pub backend: Backend,
}
impl AppState {
    pub fn new(active: usize, queue: usize, backend: Backend) -> Self {
        assert!(active > 0);
        Self {
            active: Arc::new(Semaphore::new(active)),
            waiting: Arc::new(Semaphore::new(queue)),
            backend,
        }
    }
    // Tokio's semaphore is FIFO. The waiting permit bounds the number of futures
    // allowed to wait, rather than allocating an unbounded channel of requests.
    async fn admit(&self, deadline: Instant) -> Result<OwnedSemaphorePermit, Failure> {
        if let Ok(p) = self.active.clone().try_acquire_owned() {
            return Ok(p);
        }
        let _waiting = self
            .waiting
            .clone()
            .try_acquire_owned()
            .map_err(|_| Failure::Overloaded)?;
        timeout_at(deadline, self.active.clone().acquire_owned())
            .await
            .map_err(|_| Failure::Deadline)?
            .map_err(|_| Failure::Unavailable)
    }
}
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub values: Vec<f32>,
    #[serde(default = "default_deadline")]
    pub deadline_ms: u64,
}
fn default_deadline() -> u64 {
    1000
}
impl Input {
    fn valid(&self) -> bool {
        self.values.len() == 4
            && self
                .values
                .iter()
                .all(|v| v.is_finite() && v.abs() <= 1000.0)
            && (1..=10000).contains(&self.deadline_ms)
    }
}
#[derive(Debug)]
pub enum Failure {
    Overloaded,
    Deadline,
    Unavailable,
}
impl IntoResponse for Failure {
    fn into_response(self) -> Response {
        let (status, error) = match self {
            Self::Overloaded => (StatusCode::TOO_MANY_REQUESTS, "overloaded"),
            Self::Deadline => (StatusCode::GATEWAY_TIMEOUT, "deadline_exceeded"),
            Self::Unavailable => (StatusCode::SERVICE_UNAVAILABLE, "no_healthy_worker"),
        };
        let mut r = (status, Json(json!({"error":error}))).into_response();
        if status == StatusCode::TOO_MANY_REQUESTS {
            r.headers_mut()
                .insert("retry-after", HeaderValue::from_static("1"));
        }
        r
    }
}
pub fn app(state: AppState) -> Router {
    Router::new()
        .route("/health", get(|| async { Json(json!({"status":"ok"})) }))
        .route("/ready", get(ready))
        .route("/infer", post(infer))
        .layer(DefaultBodyLimit::max(4096))
        .layer(middleware::from_fn(request_context))
        .with_state(state)
}
#[derive(Clone)]
struct Context {
    id: String,
    started: Instant,
}
async fn request_context(mut request: axum::extract::Request, next: Next) -> Response {
    let id = Uuid::new_v4().to_string();
    request.extensions_mut().insert(Context {
        id: id.clone(),
        started: Instant::now(),
    });
    // Bounds slow body reads as well. Client deadline is applied from this same
    // entry timestamp after JSON extraction, not restarted after queueing.
    let mut response =
        match timeout_at(Instant::now() + Duration::from_secs(10), next.run(request)).await {
            Ok(r) => r,
            Err(_) => Failure::Deadline.into_response(),
        };
    response
        .headers_mut()
        .insert(HeaderName::from_static("x-request-id"), id.parse().unwrap());
    response
}
async fn ready(State(state): State<AppState>) -> Response {
    match &state.backend {
        Backend::Simulated { .. } => {
            Json(json!({"backend":"simulated","ready":true})).into_response()
        }
        Backend::Workers(pool) => {
            let workers = pool.snapshot();
            let ready = workers.iter().any(|w| w.healthy);
            (
                if ready {
                    StatusCode::OK
                } else {
                    StatusCode::SERVICE_UNAVAILABLE
                },
                Json(json!({"backend":"onnx","ready":ready,"workers":workers})),
            )
                .into_response()
        }
    }
}
async fn infer(
    State(state): State<AppState>,
    axum::Extension(context): axum::Extension<Context>,
    input: Result<Json<Input>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let Ok(Json(input)) = input else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error":"invalid_json"})),
        )
            .into_response();
    };
    if !input.valid() {
        return (StatusCode::BAD_REQUEST, Json(json!({"error":"expected four finite values in [-1000,1000] and deadline_ms in [1,10000]"}))).into_response();
    }
    let deadline = context.started + Duration::from_millis(input.deadline_ms);
    let work = async {
        let _permit = state.admit(deadline).await?;
        match &state.backend {
            Backend::Simulated { delay, fail } => {
                tokio::time::sleep(*delay).await;
                if *fail {
                    return Err(Failure::Unavailable);
                }
                Ok(
                    json!({"request_id":context.id,"backend":"simulated","worker_id":null,"output":[input.values.iter().sum::<f32>()]}),
                )
            }
            Backend::Workers(pool) => {
                let result = pool.infer(&input.values, &context.id, deadline).await?;
                Ok(
                    json!({"request_id":context.id,"backend":"onnx","worker_id":result.worker_id,"output":result.output}),
                )
            }
        }
    };
    match timeout_at(deadline, work).await {
        Ok(Ok(body)) => Json(body).into_response(),
        Ok(Err(error)) => error.into_response(),
        Err(_) => Failure::Deadline.into_response(),
    }
}
/// A monitor is explicitly started by the server, never as a side effect of app construction.
pub fn start_monitor(pool: Arc<Pool>) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            pool.probe().await;
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    })
}
