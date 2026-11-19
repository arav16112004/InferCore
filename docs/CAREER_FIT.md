# Career fit and honest positioning

InferCore complements the existing resume rather than fixing a lack of projects.
The existing experience already demonstrates shipped backend/full-stack work,
PostgreSQL tuning, caching, CI/CD and production observability. Existing projects
cover RAG/MCP, a hiring platform and multimodal model training/managed serving.
The less explicit signal is ownership of request concurrency, admission control,
process failures, deadline propagation and reproducible systems measurements.
This project supplies concrete code and experiments for those conversations.

For backend SWE roles, emphasize concurrency correctness, HTTP failure semantics,
queue bounds, automated fault tests and measurements. For AI infrastructure and
ML platform roles, emphasize CPU model serving and the control plane/data plane
boundary. For AI application roles it adds useful operational depth. For modeling
and research roles it adds little evidence of novel modeling or rigorous model
experimentation; the deliberately tiny classifier is not the selling point.
No project can guarantee new-grad interviews or compensate for weak interview
fundamentals. This is a strong targeted complement, not objectively the best
project for every SWE or AI role.

Consider replacing one overlapping application project rather than adding a
fourth dense entry. Keep the multimodal project when applying for model-centric
roles; keep the strongest product project when applying for full-stack roles.
If InferCore earns a place, use two bullets that you can defend line by line.
The resume is already dense: shortening the summary and long coursework/tool
lists can improve readability without inventing additional impact. This review
uses the supplied resume as reference; its claims were not independently audited.

## Evidence-supported bullets

- Built a Rust/Tokio/Axum inference gateway routing to three isolated Python/ONNX
  Runtime CPU workers, with round-robin/least-in-flight scheduling, bounded
  admission, end-to-end deadlines and one safe retry to a different worker.
- Verified overload handling, cancellation cleanup and worker crash/recovery with
  15 automated tests; recorded successful-request p50/p95/p99, throughput, errors
  and rejections across 45,500 measured requests in matched three-worker routing-policy benchmarks.

Test-count evidence: 2 Rust unit + 5 Rust gateway + 8 Python tests. Performance
numbers belong in a bullet only with the machine, workload and concurrency
available in the report. Do not claim an improvement merely because one short
sample is faster. Do not describe this as production-scale, a distributed cluster,
LLM serving, model acceleration, exactly-once execution or zero-downtime failover.
Docker configuration exists but native results are the verified evidence.

Before interviewing, explain why deadlines include queues, what survives an HTTP
timeout, how a guard releases capacity, why retries duplicate work, why least-
in-flight is only a proxy for load, why a health check can miss a stuck model,
and what happens if the gateway itself crashes. Running generated code is not a
substitute for understanding these answers.
