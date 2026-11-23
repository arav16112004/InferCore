# Local verification — 2026-11-19

Native host: Apple M1, 8 logical CPUs, 8 GiB RAM, arm64, macOS 26.6.2.
Python 3.10.14, Rust 1.91.1. Exact resolved packages: Cargo.lock and requirements.lock.

| Check | Observed result |
| --- | --- |
| Simulated gateway before workers | 3 tests passed; live curl health and inference 200 |
| Final `cargo test --locked` | 2 routing unit tests + 5 gateway integration tests passed |
| `cargo fmt --check` | Passed |
| `cargo clippy --all-targets --locked -- -D warnings` | Passed |
| `cargo build --release --locked` | Passed |
| `.venv/bin/python -m pytest -q tests/test_system.py` | 8 passed, 15.70 seconds |
| `docker compose config --quiet` | Passed |
| Docker image build / execution | Not run: Docker Desktop daemon unavailable |
| Recorded native crash demo | Completed; see demo.json and demo-terminal.log |
| Default matched native benchmarks | 18 runs; 45,500 measured requests, all 200 |
| Separate real overload benchmark | Two 3-second runs: 1,959 successes, 2,924 HTTP 429s, zero other errors |
| Python dependency resolution | `pip check`: no broken requirements |

Rust tests exercise: validation, body bound, untrusted client IDs, unique server
IDs, FIFO queue capacity, overload, health responsiveness under inference load,
queue deadline expiration, cancellation cleanup, backend error cleanup, timeout
cleanup, routing choice, worker exclusion and lease cleanup.

Python tests exercise: deterministic trained ONNX export, numerical equivalence
against 20 held-out sklearn predictions, real three-worker routing with both
policies, concurrent requests while SIGKILL terminates a worker, unhealthy-worker
exclusion, restarting it on the same port and observing re-entry, disconnect/
malformed-output/stalled-response retries to a different worker, original-deadline
timeout, permit release, and subsequent success. A controlled blocking-session test
checks that cancellation leaves the worker's compute lock held until native work
finishes; the worker's health endpoint remains responsive and a third job is 429.

Real process tests restart the killed child explicitly; the separate recorded demo
exercises automatic restart by scripts/local.py. HTTP fault peers in tests are
controlled fakes, not used as benchmark backends. Tests use relaxed wall-clock
bounds for scheduling noise and are not formal proofs of race freedom.

The demo recorded 24 concurrent successes, 80 successes during the failure burst,
and 12 post-recovery successes. Worker 9001 was killed at t=1.759 s; it was observed
unhealthy at t=1.823 s and healthy with a new PID at t=2.951 s. These are single-run
observations, not guarantees of detection/recovery time or zero lost requests.

## Optimized release — 2026-11-22

Seven Rust tests and 13 Python tests passed against the optimized native stack. Checks included release compilation, format, strict Clippy, Python package consistency, deterministic reproduction of the model selection report, and Docker Compose configuration validation. Docker Desktop's daemon was unavailable for a container run.

New regression cases cover worker queue expiry before native computation, healthy-worker 429 and 504 handling with and without a successful alternate, and the complete remaining budget for a sole configured worker. The real-process tests check all three timing headers, an actual worker kill and return to rotation. HTTP cancellation keeps a job counted until its native future completes.

The release comparison recorded and verified every raw HTTP sample. See [method, exclusions and results](OPTIMIZATION.md). A live native smoke test sent the same input to all three workers. Each returned approximately [0.0018481, 0.9981519] with its own request ID and timing headers.
