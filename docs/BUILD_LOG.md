# Verified milestones

## 1 — Simulated gateway, 2026-11-19

Empty workspace; Python 3.10.14; Rust absent; Docker 28.4.0 client,
daemon unavailable. Installed Rust 1.91.1 inside ignored `.tools/`.

Before adding workers: `cargo test` passed 3 integration tests covering
validation/request IDs, timeout permit release, overload and health under load.
Live `cargo run` + curl: `/health` returned 200; `/infer` returned 200,
`backend: simulated`, output 10 for [1,2,3,4], and a UUID request header.
This was a 25 ms Tokio sleep, not inference. App construction lived in lib.rs;
startup in main.rs. Later milestones replace this implementation.
