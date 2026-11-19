use axum::{extract::State, http::StatusCode, response::{IntoResponse, Response}, routing::{get, post}, Json, Router};
use serde::Deserialize;
use serde_json::json;
use std::{sync::Arc, time::Duration};
use tokio::{sync::Semaphore, time::{sleep, timeout}};
use uuid::Uuid;

#[derive(Clone)]
pub struct AppState { slots: Arc<Semaphore> }
impl AppState {
    pub fn new(capacity: usize) -> Self { Self { slots: Arc::new(Semaphore::new(capacity)) } }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input { values: Vec<f32>, #[serde(default = "default_deadline")] deadline_ms: u64 }
fn default_deadline() -> u64 { 1000 }

pub fn app(state: AppState) -> Router {
    Router::new().route("/health", get(|| async { Json(json!({"status":"ok"})) }))
        .route("/infer", post(infer)).with_state(state)
}
async fn infer(State(state): State<AppState>, input: Result<Json<Input>, axum::extract::rejection::JsonRejection>) -> Response {
    let id = Uuid::new_v4().to_string();
    let reply = |status, body: serde_json::Value| {
        let mut r = (status, Json(body)).into_response();
        r.headers_mut().insert("x-request-id", id.parse().unwrap()); r
    };
    let Ok(Json(input)) = input else { return reply(StatusCode::BAD_REQUEST, json!({"error":"invalid_json","request_id":id})); };
    if input.values.len() != 4 || !input.values.iter().all(|v| v.is_finite()) || !(1..=10000).contains(&input.deadline_ms) {
        return reply(StatusCode::BAD_REQUEST, json!({"error":"invalid_input","request_id":id}));
    }
    let Ok(_permit) = state.slots.try_acquire() else {
        return reply(StatusCode::TOO_MANY_REQUESTS, json!({"error":"overloaded","request_id":id}));
    };
    match timeout(Duration::from_millis(input.deadline_ms), sleep(Duration::from_millis(25))).await {
        Ok(_) => reply(StatusCode::OK, json!({"request_id":id,"backend":"simulated","output":input.values.iter().sum::<f32>()})),
        Err(_) => reply(StatusCode::GATEWAY_TIMEOUT, json!({"error":"deadline_exceeded","request_id":id})),
    }
}
