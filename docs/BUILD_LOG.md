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

## 2 — Three real workers, bounded scheduler and fault tests

Added active and waiting semaphores, rotating round-robin and least-in-flight
selection, RAII worker leases, shared monotonic deadlines, two-attempt pure
inference retry, concurrent health probes with stale-probe protection, bounded
Python compute queues and a separate restart supervisor. A synthetic model was
used during implementation, then replaced with a trained binary Iris logistic
classifier to match the release checklist. No synthetic model is benchmarked.

Verified 7 Rust and 8 Python tests, strict Clippy and formatting; Compose config
validates. Recorded an actual SIGKILL/restart/routing-recovery demo. Native only;
Docker daemon unavailable. See VERIFICATION.md for coverage and boundaries.

## 3 — Measured routing comparison

18 native runs: concurrency 1/12/48, three repeats per policy, one-second warmup
and five-second measurements, three real ONNX workers. Preserved 45,500 measured
request records and full environment metadata. All default-suite requests were
200, so this suite alone does not demonstrate admission saturation. No reliable
policy speed advantage is claimed. A separate constrained-admission experiment
measures overload rejection with real inference. See BENCHMARKS.md.
