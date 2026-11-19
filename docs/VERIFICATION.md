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
