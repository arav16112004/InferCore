# One focused week: build, inspect, explain

The implementation is available so you can run it, but understanding requires
changing it and predicting the result. Keep each experiment and its observations
in a small commit. The first Git commit contains only the simulated gateway;
`git show 545c9d3:src/lib.rs` lets you study it without losing the current code.

## Day 1: ownership, async, HTTP

Read src/lib.rs and src/main.rs. `app(state)` constructs a Router without binding
a port; Tower `oneshot` tests send in-memory HTTP requests to the same router.
Startup owns the listener and shutdown policy. This separation avoids tests
requiring a production server process just to validate a handler.

A Rust value has one owner. Moving a semaphore permit into a future means that
future owns the right to use capacity. When it exits or is dropped, Rust drops
the permit and returns capacity. `Arc::clone` increments a shared-reference count;
it does not clone the entire scheduler. `Arc` alone does not allow mutation:
metadata needs a mutex. A borrowed reference cannot outlive its owner, which is
why independent spawned tasks often need owned values and `Arc` handles.

An `async fn` returns a future. `.await` can suspend that future without blocking
an OS thread. Tokio polls ready futures; CPU-heavy code that never yields prevents
other work on that executor thread. Waiting concurrently is different from
performing CPU arithmetic in parallel. Python ONNX native work runs outside the
HTTP event loop in a single-thread executor. Each worker can have one call
computing and one waiting for that thread.

Experiment: run simulated mode, issue parallel requests and `/health`. Predict
what happens if the async sleep becomes a blocking thread sleep, then verify in a
temporary branch. Explain why increasing Tokio threads does not make one model
invocation intrinsically faster. Run the gateway tests before and after.

## Day 2: process pool and shared state

Read src/pool.rs, worker/server.py and scripts/local.py. The gateway shares counts
within one process; workers do not share Python state. Each loads its own session.
Observe three distinct PIDs with the worker health endpoints and distinct worker
IDs with nine sequential requests.

Round-robin distributes selections; least-in-flight responds to outstanding work.
Neither knows the actual cost of the next request. The mutex covers selecting and
incrementing together. Holding it during HTTP would serialize requests and delay
health state updates, so network waits happen after the lock is released.

Exercise: explain how two simultaneous requests could overload one worker if the
selection and increment were separate critical sections. Read the Lease Drop
implementation and identify all return/cancellation paths it protects.

## Day 3: queueing and real inference

Regenerate the trained Iris model and inspect its SHA-256. The ONNX graph really
executes trained coefficients. Training is offline; serving loads a fixed artifact.
The model uses four numbers and two probabilities; HTTP/JSON dominates its cost.
A larger model is a later workload experiment, not a prerequisite for correctness.

Admission has two resources: active slots N and waiting slots Q. Waiting for a
Tokio semaphore is the queue; Tokio's FIFO ordering determines who proceeds.
Only Q futures may wait for active capacity. Queueing hides short bursts but adds
latency; if arrival rate persistently exceeds service rate, any finite queue fills.
Little's law relates average in-system requests, arrival throughput and average
latency in a stable system; it does not guarantee a tail percentile.

Experiment: simulated mode with MAX_IN_FLIGHT=1 and QUEUE_CAPACITY=0, then 1, then
12. Send a burst and record successes, rejections and latency. Predict why the
larger queue can reduce immediate 429s while increasing p99 and 504s. Identify the
second, bounded queue inside each worker and the cost of buffering in two places.

## Day 4: deadlines, cancellation and failure

Read the middleware timestamp, admit(), Pool::infer(), health epochs and worker
shielded task. One absolute deadline prevents queue wait plus retry from silently
exceeding the client's total budget. Independent per-stage one-second timeouts
would add up to several seconds. The half-budget first attempt leaves retry room
when several workers are configured. A single configured worker gets the full budget.

Run scripts/demo.py. Kill the worker, observe failed transport, exclusion, a new
PID and successful routing back to that worker. Distinguish the supervisor's
process restart from the gateway monitor's re-admission. Probe cadence is not a
hard recovery-time guarantee. A healthy process can still have a stuck compute
thread, and transport timeout does not prove the operation never ran.

Explain why retrying pure inference is safe for correctness but still uses CPU.
Then explain why the same retry policy is unsafe for charging a credit card. Read
the cancellation test and predict what would happen if HTTP cancellation returned
a worker slot while its submitted job was still running.

## Day 5: measurements

Run scripts/benchmark.py using identical worker count, concurrency, inputs,
deadline, warmup and measurement duration for each policy. Read manifest.json,
summary.json and one raw JSONL. p50 describes typical successful latency, p95/p99
the slower tail. A policy with lower p95 and many failed requests may be worse.
Inspect counts and success throughput alongside latency. Fast 429s must never
be mixed into successful inference percentiles.

The supplied comparison is closed-loop and same-host. Explain coordinated
omission: stalled clients stop sending, hiding the queue an external fixed-rate
arrival process would create. Repeat longer runs before making a performance
claim. Consider larger trained models and controlled heterogeneous service times
before batching. Batching adds waiting and changes the API/workload; it is deferred.

## Day 6: defend the design

Draw the README request flow from memory. Explain what each capacity count means,
where deadlines apply, what Drop releases, what keeps running after timeout, and
which failures return 429/503/504. Identify the single gateway as a single point of
failure. Explain how several gateways would have inconsistent worker counts, and
why this implementation does not solve distributed coordination or consensus.

Write down two alternatives for each choice: thread vs process, HTTP vs IPC,
semaphore wait queue vs dispatcher channel, round-robin vs least-in-flight,
fail-fast vs retry, restarting workers vs cancelling individual calls. State the
problem each solves and the complexity it introduces. No dashboard is needed to
answer these questions; the raw evidence matters more.

## Day 7: polish and demonstrate

Run the documented setup, tests, demo and benchmark from a fresh clone. Review the
public repository for private files and inaccurate claims. The resume PDF is never
part of this repository. Use evidence-linked bullets from CAREER_FIT.md. Practice
a five-minute explanation plus a crash demo. Optional open-source work should be a
small, understood, independently tested contribution; a rushed unrelated PR is
not a release requirement. A terminal transcript is recorded; no video is claimed.

Next project, if your goal is deeper distributed systems: a replicated key-value
service or implementing Raft in a learning environment. That teaches leader
failure, log replication, consistency and network partitions that InferCore does
not cover. Do this after you can defend the serving system's concurrency model.
