# Native benchmark results — 2026-11-19

Measured on Apple M1, 8 logical CPUs, 8 GiB RAM, macOS 26.6.2 arm64.
Rust 1.91.1 release build; Python 3.10.14; ONNX Runtime 1.20.1, CPU provider.
Three independent workers; one ORT intra-op/inter-op thread each; one compute
slot and two pending jobs per worker. Gateway: 6 active requests, 12 waiters.
Model: trained binary Iris logistic regression, 319-byte ONNX, opset 17/IR 9.

Workload: `[6.7,3.0,5.2,2.3]`, deadline 1000 ms, HTTP keepalive, no think
time, HTTPX closed-loop client on the same host. Each policy/concurrency has
3 repeats, each with 1 s warmup and 5 s measurement.
Policy order reverses on alternating repeats. Timings include request/response
serialization, loopback transport, gateway/worker queues and client overhead.

The table reports the median of per-run rates/percentiles (not a pooled latency
percentile). Errors and rejections are totals across the repeats. All latency
columns include successful responses only. Full per-status latency distributions,
histograms and every request are preserved in the linked raw evidence.

| Concurrency | Policy | Median successes/s | p50 ms | p95 ms | p99 ms | 429 total | Other errors total |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | round-robin | 703.9 | 1.38 | 1.67 | 2.03 | 0 | 0 |
| 1 | least-in-flight | 680.0 | 1.38 | 1.78 | 2.22 | 0 | 0 |
| 12 | round-robin | 627.6 | 10.94 | 13.10 | 19.82 | 0 | 0 |
| 12 | least-in-flight | 631.0 | 10.55 | 12.34 | 21.80 | 0 | 0 |
| 48 | round-robin | 203.5 | 111.97 | 856.65 | 1591.53 | 0 | 0 |
| 48 | least-in-flight | 201.2 | 113.03 | 832.82 | 1578.47 | 0 | 0 |

## Interpretation and limits

These are short exploratory samples. They do not establish a statistically
reliable routing-policy advantage, a throughput improvement over another system,
or production capacity. Identical workers and this tiny model limit the value of
least-in-flight routing; counts are not a prediction of remaining service time.
At higher concurrency, the same-host Python client and HTTP overhead can dominate.
Successful throughput can decrease while total responses increase because 429s
are cheap. Read error/rejection counts before comparing successful latency.

Closed-loop clients stop offering work while stalled (coordinated omission).
There is no independent offered arrival rate, so this does not characterize
latency under sustained open-loop overload. Run longer samples and a separate
load-generator host before drawing stronger conclusions. No batching, artificial
delays, simulated backends or worker failures are included in these measurements.

## Raw evidence and reproduction

- [Hardware, dependency versions, model hash, source hashes and workload](../benchmarks/m1-native/manifest.json)
- [Per-run summaries, status counts and latency histograms](../benchmarks/m1-native/summary.json)
- [Every request as JSONL](../benchmarks/m1-native/)

```sh
source scripts/env.sh  # original project-local Rust installation only
cargo build --release --locked
.venv/bin/python scripts/benchmark.py --output benchmarks/new-run
.venv/bin/python scripts/benchmark_report.py benchmarks/new-run
```

The manifest records the implementation commit and SHA-256 of the binary and
source files. Documentation/reporting commits after that do not change the
measured implementation. Runtime packages are recorded in both lockfiles and
the manifest. No Docker result is claimed.

Client-observed p99 at concurrency 48 exceeds the 1000 ms gateway budget.
Client timing starts before connection/pool/transport waiting; the gateway budget
starts at middleware receipt. These are different intervals. The data does not
localize the slowdown: per-stage tracing and a separate client would be needed.
Measured requests: 45,500; overload rejections: 0;
other errors: 0. Consult counts to assess admission saturation.

## Separate real-inference overload experiment

Same model, three workers and concurrency 12, but **one active request and
zero queue slots**. One 3-second sample per policy after 1-second warmup.
These limits differ from the default suite: do not compare them as a policy
optimization. A single sample cannot support a scheduling improvement claim.

| Policy | Successes | 429s | Other errors | Successes/s | Success p50/p95/p99 ms |
| --- | ---: | ---: | ---: | ---: | --- |
| round-robin | 815 | 2095 | 0 | 270.9 | 9.95 / 51.71 / 94.75 |
| least-in-flight | 1144 | 829 | 0 | 380.0 | 10.36 / 13.54 / 74.97 |

[Separate metadata](../benchmarks/m1-overload/manifest.json),
[summary and latency for rejected requests](../benchmarks/m1-overload/summary.json),
[raw requests](../benchmarks/m1-overload/).

```sh
.venv/bin/python scripts/benchmark.py --output benchmarks/new-overload --max-in-flight 1 --queue-capacity 0 --concurrency 12 --repeats 1 --seconds 3 --warmup 1
```
