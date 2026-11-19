#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:8080").await?;
    println!("InferCore: http://127.0.0.1:8080 (simulated backend)");
    axum::serve(listener, infercore::app(infercore::AppState::new(2))).await?;
    Ok(())
}
