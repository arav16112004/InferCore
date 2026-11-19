use axum::{body::Body, http::{Request, StatusCode}};
use infercore::{app, AppState};
use tower::ServiceExt;
fn request(deadline: u64) -> Request<Body> {
    Request::post("/infer").header("content-type", "application/json")
        .body(Body::from(format!(r#"{{"values":[1,2,3,4],"deadline_ms":{deadline}}}"#))).unwrap()
}
#[tokio::test]
async fn simulated_success_and_validation() {
    let a = app(AppState::new(1));
    let r = a.clone().oneshot(request(1000)).await.unwrap();
    assert_eq!(r.status(), StatusCode::OK);
    assert!(r.headers().contains_key("x-request-id"));
    let r = a.oneshot(Request::post("/infer").header("content-type", "application/json").body(Body::from("{}")).unwrap()).await.unwrap();
    assert_eq!(r.status(), StatusCode::BAD_REQUEST);
}
#[tokio::test]
async fn timeout_releases_capacity() {
    let a = app(AppState::new(1));
    assert_eq!(a.clone().oneshot(request(1)).await.unwrap().status(), StatusCode::GATEWAY_TIMEOUT);
    assert_eq!(a.oneshot(request(1000)).await.unwrap().status(), StatusCode::OK);
}
#[tokio::test]
async fn overload_and_health_during_load() {
    let a = app(AppState::new(1));
    let running = tokio::spawn(a.clone().oneshot(request(1000)));
    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    assert_eq!(a.clone().oneshot(request(1000)).await.unwrap().status(), StatusCode::TOO_MANY_REQUESTS);
    let health = a.oneshot(Request::get("/health").body(Body::empty()).unwrap());
    assert_eq!(tokio::time::timeout(std::time::Duration::from_millis(10), health).await.unwrap().unwrap().status(), StatusCode::OK);
    assert_eq!(running.await.unwrap().unwrap().status(), StatusCode::OK);
}
