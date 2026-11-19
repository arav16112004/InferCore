use infercore::{
    app,
    pool::{Backend, Policy, Pool},
    start_monitor, AppState,
};
use std::{env, time::Duration};
fn number(name: &str, default: usize) -> Result<usize, Box<dyn std::error::Error>> {
    Ok(env::var(name)
        .unwrap_or_else(|_| default.to_string())
        .parse()?)
}
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let backend_name = env::var("BACKEND").unwrap_or_else(|_| "simulated".into());
    let mut monitor = None;
    let backend = match backend_name.as_str() {
        "simulated" => Backend::Simulated {
            delay: Duration::from_millis(25),
            fail: false,
        },
        "onnx" => {
            let urls = env::var("WORKER_URLS").unwrap_or_else(|_| {
                "http://127.0.0.1:9001,http://127.0.0.1:9002,http://127.0.0.1:9003".into()
            });
            let urls = urls
                .split(',')
                .map(|s| s.trim().trim_end_matches('/').to_string())
                .collect::<Vec<_>>();
            for url in &urls {
                let parsed = reqwest::Url::parse(url)?;
                if parsed.scheme() != "http" || parsed.host_str().is_none() {
                    return Err("worker URLs must be HTTP URLs".into());
                }
            }
            let policy: Policy = env::var("POLICY")
                .unwrap_or_else(|_| "round-robin".into())
                .parse()?;
            let pool = Pool::new(urls, policy, 2);
            pool.probe().await;
            monitor = Some(start_monitor(pool.clone()));
            Backend::Workers(pool)
        }
        _ => return Err("BACKEND must be simulated or onnx".into()),
    };
    let active = number("MAX_IN_FLIGHT", 6)?;
    if active == 0 {
        return Err("MAX_IN_FLIGHT must be positive".into());
    }
    let state = AppState::new(active, number("QUEUE_CAPACITY", 12)?, backend);
    let address = env::var("BIND").unwrap_or_else(|_| "127.0.0.1:8080".into());
    let listener = tokio::net::TcpListener::bind(&address).await?;
    println!("InferCore http://{address}, backend={backend_name}");
    axum::serve(listener, app(state))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    if let Some(task) = monitor {
        task.abort();
    }
    Ok(())
}
