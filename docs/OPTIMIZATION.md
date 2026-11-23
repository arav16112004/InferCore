# Optimizing InferCore on Apple M1

The release comparison measures the original gateway and worker code at `2de5cc2` against the optimized stack. Both stacks receive the **same ONNX model bytes**, the same six rotating inputs and the same deadline. The model-quality experiment below answers a different question using labeled flowers.

## Serving measurements

On this Apple M1 with 12 closed-loop callers and three workers, round-robin delivered **4,749 → 7,782 successful predictions/s**, with p95 **4.327 → 2.376 ms**. All requests in those included runs succeeded. At one caller routed through a single worker, p95 fell from 0.941 to 0.472 ms. Worker startup to first observed ready response had medians of **1,156 ms before and 761 ms after**, across three sequential starts per version. Startup includes polling.

The client recorded **847,175** HTTP requests. Seven nominal two-second runs lasted from roughly 72 to 1,018 seconds. Both members of each affected matched pair are excluded wherever this report compares the versions. The exclusion rule is elapsed time exceeding the configured measurement window by 30 seconds. The remaining 50 runs contain **595,599** measured requests; all finished their two-second window and drained within 2.010 seconds. There are 2–4 included paired repeats per scenario (three for round-robin with 12 clients). Raw files and summaries retain every run. The verification command below recomputes their counts, rates and timing percentiles before applying the exclusion rule.

| Path | Clients | Version | OK/s | p50 ms | p95 ms | p99 ms | OK | 429 | Other errors |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| worker | 1 | baseline | 2,021 | 0.460 | 0.613 | 0.790 | 11,737 | 0 | 0 |
| worker | 1 | optimized | 3,490 | 0.221 | 0.334 | 0.415 | 22,459 | 0 | 0 |
| gateway1 | 1 | baseline | 1,261 | 0.775 | 0.941 | 1.358 | 9,809 | 0 | 0 |
| gateway1 | 1 | optimized | 2,506 | 0.393 | 0.472 | 0.540 | 18,752 | 0 | 0 |
| worker | 12 | baseline | 196 | 12.624 | 23.014 | 28.408 | 1,283 | 23,877 | 0 |
| worker | 12 | optimized | 400 | 3.192 | 25.236 | 46.461 | 4,067 | 51,686 | 0 |
| gateway1 | 12 | baseline | 1,820 | 6.541 | 9.949 | 12.581 | 10,783 | 0 | 0 |
| gateway1 | 12 | optimized | 3,372 | 2.517 | 8.701 | 14.213 | 21,044 | 0 | 0 |
| round-robin | 12 | baseline | 4,749 | 2.115 | 4.327 | 9.621 | 23,406 | 0 | 0 |
| round-robin | 12 | optimized | 7,782 | 1.432 | 2.376 | 3.330 | 42,236 | 0 | 0 |
| least-in-flight | 12 | baseline | 2,225 | 5.314 | 6.923 | 8.768 | 17,471 | 0 | 0 |
| least-in-flight | 12 | optimized | 3,733 | 3.192 | 4.076 | 5.176 | 28,877 | 0 | 0 |
| round-robin | 48 | baseline | 2,251 | 13.717 | 18.572 | 22.459 | 16,010 | 72,177 | 0 |
| round-robin | 48 | optimized | 3,856 | 9.269 | 11.572 | 14.748 | 32,617 | 45,522 | 0 |
| least-in-flight | 48 | baseline | 2,890 | 9.069 | 11.489 | 15.232 | 11,590 | 51,796 | 0 |
| least-in-flight | 48 | optimized | 6,850 | 4.089 | 7.210 | 15.966 | 27,427 | 50,973 | 0 |

**Where time remains.** Direct, warmed ONNX calls on preallocated inputs had median p50 around 4.4 µs. Through the optimized round-robin gateway at 12 callers, the median-of-run p50 was 509 µs for gateway admission, 86 µs for executor dispatch/queueing, and 20 µs for the ONNX call. These are different stages and measurement contexts; their percentiles cannot be added into an end-to-end percentile. Gateway admission was the largest of the three instrumented stages at this load. Raising the queue limit would let more requests wait; it would not create more compute slots.

**Tradeoff at saturation.** Direct worker HTTP at 12 callers showed higher successful throughput but worse successful p99 (28.408 → 46.461 ms). Both versions rejected many calls with 429. Least-in-flight at 48 callers also had a slightly higher p99 in the two retained pairs. The short samples and substantial host variability limit cross-policy conclusions. Choose capacity and rejection targets with a representative model and arrival rate before changing the defaults.

The throughput columns count only HTTP 200 responses whose two probabilities match the direct ONNX reference within 0.000002. The client also checks the gateway response ID and worker ID. Latencies are measured at the client and include response validation. Rates and percentiles are medians of per-run measurements. Status counts are summed over the included runs. A closed-loop caller sends its next request as soon as the previous one completes, including after rejection. At high concurrency this can generate a large number of quick 429s. Compare useful throughput, success latency **and** rejection rates together.

- `worker` sends HTTP directly to one worker, which can admit two jobs.
- `gateway1` sends the same workload through a gateway with one configured worker, six active requests and twelve gateway queue places. Queuing and HTTP overhead are both part of this path.
- `round-robin` and `least-in-flight` exercise the normal three-worker deployment. Each worker runs one ONNX call at a time and may accept one additional job. All three use the CPU provider and one intra-op and one inter-op thread.

## Changes behind the measurements

1. **Process startup.** A serving worker previously imported scikit-learn, SciPy and the ONNX training/export stack to get a file path. The worker now imports that path from a small module. The Docker worker uses a runtime lockfile. The full development install includes training and benchmark tools.
2. **Per-request work in Python.** Originally the event loop created a shielded task that acquired an async lock before submitting the job to a thread. A dedicated single-thread executor now serializes the two admitted jobs. The event loop checks capacity before submission and releases it when the native future finishes. A disconnect or gateway timeout cannot make an executing worker slot look free. The response is encoded once. Pinned uvloop and httptools accelerate the worker event loop and HTTP parser on the measured Mac.
3. **Worker selection in Rust.** The previous path allocated an eligible-worker list, sorted it for least-in-flight, copied pool status to get the URL and allocated a set to exclude one worker. A single rotating scan now reserves the worker under the pool mutex. The URL is prebuilt, the excluded ID is optional, and the worker response decodes directly into a two-element array. No mutex spans a network wait.
4. **Overload and deadlines.** An explicit worker 429 or worker-side queued-job 504 leaves a responsive worker healthy. One other eligible worker can be tried within the original deadline. A single configured worker gets the entire remaining budget. Connection faults, malformed output, or an unresponsive peer still trigger unhealthy-worker exclusion.
5. **Request timing.** Three success headers report gateway admission, executor dispatch/queue wait, and the ONNX call in integer microseconds. The last two describe only the successful worker attempt. Their sum excludes HTTP parsing, serialization, socket waiting, worker selection and any earlier failed attempt. A successful prediction can have practically zero gateway queue wait and still spend time elsewhere.

## Classifier optimization

The original classifier used unscaled features and C=1. On the original 80 training flowers, we compared six combinations of scaling and C (0.1, 1, 10) by validation log loss. The final five-fold selection chose scaling and C=10. Standardization is fitted only on the fit rows of each fold.

To estimate the selection procedure on unseen rows, the script repeated stratified five-fold outer validation with three seeds, choosing the settings separately by four-fold inner validation inside each outer training split. The unchanged baseline and the selected procedure scored the same outer rows. Average log loss fell from **0.2201 to 0.1262**. Both got **227/240** outer predictions correct. These are repeated evaluations of 80 flowers. They are not 240 independent examples. The data supports an improvement in probability scores on these folds, with no observed gain in label accuracy.

The export absorbs standardization into the classifier: `w_raw = w_scaled / scale` and `b_raw = b_scaled - dot(w_raw, mean)`. Workers still accept centimeters; the ONNX graph still has five operators and occupies 319 bytes. The previously observed 20-row holdout scored 20/20. It was excluded from parameter selection. A parity test compares the trained scikit-learn pipeline and ONNX outputs on every holdout row. See [folds, candidate scores and selection](../models/selection.json).

## Reproduce

On macOS use the project's Python 3.10 environment, pinned Rust toolchain and Node.js. Node's built-in HTTP client uses one keep-alive connection per caller. The paired comparison launches its own workers and gateways on ephemeral loopback ports and checks child identity before sending traffic. For the recorded run the original binary also matched the SHA-256 from the first benchmark manifest.

```sh
# From the optimized checkout, with Cargo and rustc on PATH
.venv/bin/python -m pip install -r requirements.lock
cargo build --release --locked
git worktree add --detach .tools/original 2de5cc2
cargo build --release --locked --manifest-path .tools/original/Cargo.toml
.venv/bin/python -m scripts.profile_paths --native --baseline .tools/original --output benchmarks/my-comparison --seconds 2 --warmup 0.5 --repeats 4
.venv/bin/python -m scripts.summarize_profile benchmarks/my-comparison --verify
.venv/bin/python -m scripts.evaluate_model
```

The direct ONNX microbenchmark warms 500 calls and then times three series of 10,000 calls with preallocated inputs. For each HTTP scenario the driver alternates baseline/optimized order across four paired repeats. The baseline uses h11/asyncio. The optimized variant uses httptools/uvloop. The Mac runs both the Node load generator and the serving stack. Short loopback measurements characterize this workload and include client CPU contention. Comparing the whole revision measures the combined benefit; it cannot assign a percentage to each code change. Older HTTPX results use different client scheduling and must not be used as the denominator for these speedups.

[Environment, exact source/binary hashes and workload](../benchmarks/release-comparison/manifest.json) · [Per-run distributions](../benchmarks/release-comparison/summary.json) · [Compressed raw requests and ONNX timings](../benchmarks/release-comparison/)
