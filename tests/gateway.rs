use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use infercore::{app, pool::Backend, AppState};
use std::time::Duration;
use tower::ServiceExt;
fn state(active: usize, queue: usize, fail: bool) -> AppState {
    AppState::new(
        active,
        queue,
        Backend::Simulated {
            delay: Duration::from_millis(80),
            fail,
        },
    )
}
fn request(deadline: u64) -> Request<Body> {
    Request::post("/infer")
        .header("content-type", "application/json")
        .body(Body::from(format!(
            r#"{{"values":[1,2,3,4],"deadline_ms":{deadline}}}"#
        )))
        .unwrap()
}
#[tokio::test]
async fn simulated_success_and_validation() {
    let a = app(state(1, 0, false));
    let r = a.clone().oneshot(request(1000)).await.unwrap();
    assert_eq!(r.status(), StatusCode::OK);
    let id = r.headers()["x-request-id"].clone();
    let r = a.clone().oneshot(request(1000)).await.unwrap();
    assert_ne!(id, r.headers()["x-request-id"]);
    for body in [
        "{}",
        r#"{"values":[1,2,3]}"#,
        r#"{"values":[1,2,3,4],"deadline_ms":0}"#,
        r#"{"values":[1,2,3,1001]}"#,
    ] {
        let r = a
            .clone()
            .oneshot(
                Request::post("/infer")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::BAD_REQUEST);
        assert!(r.headers().contains_key("x-request-id"));
    }
}
#[tokio::test]
async fn timeout_and_error_release_capacity() {
    let a = app(state(1, 0, false));
    assert_eq!(
        a.clone().oneshot(request(1)).await.unwrap().status(),
        StatusCode::GATEWAY_TIMEOUT
    );
    assert_eq!(
        a.oneshot(request(1000)).await.unwrap().status(),
        StatusCode::OK
    );
    let a = app(state(1, 0, true));
    for _ in 0..2 {
        assert_eq!(
            a.clone().oneshot(request(1000)).await.unwrap().status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
    }
}
#[tokio::test]
async fn bounded_queue_overload_and_health_during_load() {
    let a = app(state(1, 1, false));
    let first = tokio::spawn(a.clone().oneshot(request(1000)));
    tokio::time::sleep(Duration::from_millis(10)).await;
    let queued = tokio::spawn(a.clone().oneshot(request(1000)));
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert_eq!(
        a.clone().oneshot(request(1000)).await.unwrap().status(),
        StatusCode::TOO_MANY_REQUESTS
    );
    let health = a.oneshot(Request::get("/health").body(Body::empty()).unwrap());
    assert_eq!(
        tokio::time::timeout(Duration::from_millis(30), health)
            .await
            .unwrap()
            .unwrap()
            .status(),
        StatusCode::OK
    );
    assert_eq!(first.await.unwrap().unwrap().status(), StatusCode::OK);
    assert_eq!(queued.await.unwrap().unwrap().status(), StatusCode::OK);
}
#[tokio::test]
async fn queue_deadline_and_cancel_release_capacity() {
    let a = app(state(1, 1, false));
    let first = tokio::spawn(a.clone().oneshot(request(1000)));
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert_eq!(
        a.clone().oneshot(request(5)).await.unwrap().status(),
        StatusCode::GATEWAY_TIMEOUT
    );
    // The expired waiter releases its queue slot; a new waiter can enter.
    let queued = tokio::spawn(a.clone().oneshot(request(1000)));
    tokio::time::sleep(Duration::from_millis(10)).await;
    first.abort();
    let _ = first.await;
    assert_eq!(queued.await.unwrap().unwrap().status(), StatusCode::OK);
    assert_eq!(
        a.oneshot(request(1000)).await.unwrap().status(),
        StatusCode::OK
    );
}
#[tokio::test]
async fn body_limit_and_untrusted_request_id() {
    let a = app(state(1, 0, false));
    let mut req = request(1000);
    req.headers_mut()
        .insert("x-request-id", "client-chosen".parse().unwrap());
    let r = a.clone().oneshot(req).await.unwrap();
    assert_ne!(r.headers()["x-request-id"], "client-chosen");
    let r = a
        .oneshot(
            Request::post("/infer")
                .header("content-type", "application/json")
                .body(Body::from(" ".repeat(5000)))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(r.status(), StatusCode::BAD_REQUEST);
}
