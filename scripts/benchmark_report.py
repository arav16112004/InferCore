"""Summarize recorded evidence only; does not rerun inference or infer speedups."""
import json
from pathlib import Path
import statistics
import sys

root=Path(__file__).resolve().parents[1]
folder=Path(sys.argv[1]) if len(sys.argv)>1 else root/"benchmarks/m1-native"
manifest=json.loads((folder/"manifest.json").read_text())
runs=json.loads((folder/"summary.json").read_text())
expected=len(manifest["concurrency"])*manifest["repeats"]*2
if len(runs)!=expected:
    raise SystemExit(f"incomplete benchmark: {len(runs)}/{expected} runs")
lines=["# Native benchmark results — 2026-11-19", "",
"Measured on Apple M1, 8 logical CPUs, 8 GiB RAM, macOS 26.6.2 arm64.",
"Rust 1.91.1 release build; Python 3.10.14; ONNX Runtime 1.20.1, CPU provider.",
"Three independent workers; one ORT intra-op/inter-op thread each; one compute",
f"slot and two pending jobs per worker. Gateway: {manifest['max_in_flight']} active requests, {manifest['queue_capacity']} waiters.",
"Model: trained binary Iris logistic regression, 319-byte ONNX, opset 17/IR 9.",
"", "Workload: `[6.7,3.0,5.2,2.3]`, deadline 1000 ms, HTTP keepalive, no think",
"time, HTTPX closed-loop client on the same host. Each policy/concurrency has",
f"{manifest['repeats']} repeats, each with {manifest['warmup_seconds']} s warmup and {manifest['measurement_seconds']} s measurement.",
"Policy order reverses on alternating repeats. Timings include request/response",
"serialization, loopback transport, gateway/worker queues and client overhead.","",
"The table reports the median of per-run rates/percentiles (not a pooled latency",
"percentile). Errors and rejections are totals across the repeats. All latency",
"columns include successful responses only. Full per-status latency distributions,",
"histograms and every request are preserved in the linked raw evidence.","",
"| Concurrency | Policy | Median successes/s | p50 ms | p95 ms | p99 ms | 429 total | Other errors total |",
"| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
for c in manifest["concurrency"]:
    for p in ("round-robin","least-in-flight"):
        rs=[r for r in runs if r["concurrency"]==c and r["policy"]==p]
        rate=statistics.median(r["success_rps"] for r in rs)
        ps=[statistics.median(r["successful_latency"][f"p{q}_ms"] for r in rs) for q in (50,95,99)]
        lines.append(f"| {c} | {p} | {rate:.1f} | {ps[0]:.2f} | {ps[1]:.2f} | {ps[2]:.2f} | {sum(r['overload_rejections'] for r in rs)} | {sum(r['errors_excluding_overload'] for r in rs)} |")
lines += ["", "## Interpretation and limits", "",
"These are short exploratory samples. They do not establish a statistically",
"reliable routing-policy advantage, a throughput improvement over another system,",
"or production capacity. Identical workers and this tiny model limit the value of",
"least-in-flight routing; counts are not a prediction of remaining service time.",
"At higher concurrency, the same-host Python client and HTTP overhead can dominate.",
"Successful throughput can decrease while total responses increase because 429s",
"are cheap. Read error/rejection counts before comparing successful latency.","",
"Closed-loop clients stop offering work while stalled (coordinated omission).",
"There is no independent offered arrival rate, so this does not characterize",
"latency under sustained open-loop overload. Run longer samples and a separate",
"load-generator host before drawing stronger conclusions. No batching, artificial",
"delays, simulated backends or worker failures are included in these measurements.","",
"## Raw evidence and reproduction", "",
"- [Hardware, dependency versions, model hash, source hashes and workload](../benchmarks/m1-native/manifest.json)",
"- [Per-run summaries, status counts and latency histograms](../benchmarks/m1-native/summary.json)",
"- [Every request as JSONL](../benchmarks/m1-native/)","",
"```sh", "source scripts/env.sh  # original project-local Rust installation only",
"cargo build --release --locked",
".venv/bin/python scripts/benchmark.py --output benchmarks/new-run",
".venv/bin/python scripts/benchmark_report.py benchmarks/new-run", "```", "",
"The manifest records the implementation commit and SHA-256 of the binary and",
"source files. Documentation/reporting commits after that do not change the",
"measured implementation. Runtime packages are recorded in both lockfiles and",
"the manifest. No Docker result is claimed."]
lines += ["", "Client-observed p99 at concurrency 48 exceeds the 1000 ms gateway budget.",
"Client timing starts before connection/pool/transport waiting; the gateway budget",
"starts at middleware receipt. These are different intervals. The data does not",
"localize the slowdown: per-stage tracing and a separate client would be needed.",
f"Measured requests: {sum(r['requests'] for r in runs):,}; overload rejections: {sum(r['overload_rejections'] for r in runs):,};",
f"other errors: {sum(r['errors_excluding_overload'] for r in runs):,}. Consult counts to assess admission saturation."]
overload=root/"benchmarks/m1-overload/summary.json"
if overload.exists():
    rs=json.loads(overload.read_text())
    lines += ["", "## Separate real-inference overload experiment", "",
    "Same model, three workers and concurrency 12, but **one active request and",
    "zero queue slots**. One 3-second sample per policy after 1-second warmup.",
    "These limits differ from the default suite: do not compare them as a policy",
    "optimization. A single sample cannot support a scheduling improvement claim.", "",
    "| Policy | Successes | 429s | Other errors | Successes/s | Success p50/p95/p99 ms |",
    "| --- | ---: | ---: | ---: | ---: | --- |"]
    for r in rs:
        q=r["successful_latency"]
        lines.append(f"| {r['policy']} | {r['status_counts'].get('200',0)} | {r['overload_rejections']} | {r['errors_excluding_overload']} | {r['success_rps']:.1f} | {q['p50_ms']:.2f} / {q['p95_ms']:.2f} / {q['p99_ms']:.2f} |")
    lines += ["", "[Separate metadata](../benchmarks/m1-overload/manifest.json),",
    "[summary and latency for rejected requests](../benchmarks/m1-overload/summary.json),",
    "[raw requests](../benchmarks/m1-overload/).", "", "```sh",
    ".venv/bin/python scripts/benchmark.py --output benchmarks/new-overload --max-in-flight 1 --queue-capacity 0 --concurrency 12 --repeats 1 --seconds 3 --warmup 1", "```"]
(root/"docs/BENCHMARKS.md").write_text("\n".join(lines).replace("../benchmarks/m1-native", "../" + str(folder.resolve().relative_to(root)))+"\n")
print("\n".join(lines))
