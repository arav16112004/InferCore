"""Check raw HTTP samples and print a side-by-side release comparison."""
import argparse
from collections import Counter
import gzip
import json
from math import isclose
from pathlib import Path
from statistics import median
import sys

from scripts.benchmark import percentile


def verify(folder, run, schema):
    with gzip.open(folder / (run["key"] + ".json.gz"), "rt") as stream:
        rows = json.load(stream)
    status = schema.index("status")
    samples = [row for row in rows if row[status] == 200]
    counts = Counter(str(row[status]) for row in rows)
    if counts != run["status"] or not isclose(len(samples) / run["seconds"], run["success_rps"]):
        raise ValueError(f"Count/rate mismatch: {run['key']}")
    for field, stat in [("latency_ms", "latency_ms"), ("admission_us", "admission_us"),
                        ("worker_queue_us", "worker_queue_us"), ("ort_run_us", "ort_run_us")]:
        col = schema.index(field)
        values = [row[col] for row in samples if row[col] is not None]
        expected = {f"p{p}": percentile(values, p / 100) for p in (50, 95, 99)}
        if expected != run[stat]:
            raise ValueError(f"Invalid {stat} distribution: {run['key']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--verify", action="store_true", help="recompute metrics from compressed raw samples")
    args = parser.parse_args()
    manifest = json.loads((args.folder / "manifest.json").read_text())
    runs = json.loads((args.folder / "summary.json").read_text())
    if args.verify:
        for run in runs:
            verify(args.folder, run, manifest["rows"])

    # A request has a 1 s gateway budget and the HTTP client a 3 s socket timeout.
    # A drain over 30 s beyond the timed window identifies a major host interruption.
    # Drop the matching repeat on both sides. Preserve and verify every raw row.
    interrupted = {(r["path"], r["concurrency"], r["repeat"]) for r in runs
                   if r["seconds"] > manifest["measurement_s"] + 30}
    if interrupted:
        print(f"Excluded matched pairs with interrupted clocks: {sorted(interrupted)}", file=sys.stderr)

    print("| Path | Clients | Version | OK/s | p50 ms | p95 ms | p99 ms | OK | 429 | Other errors |")
    print("| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for path, clients in dict.fromkeys((run["path"], run["concurrency"]) for run in runs):
        for version in ("baseline", "optimized"):
            group = [r for r in runs if (r["path"], r["concurrency"], r["variant"]) == (path, clients, version)]
            if sorted(r["repeat"] for r in group) != list(range(1, manifest["repeats"] + 1)):
                raise ValueError(f"Incomplete group: {path}, {clients}, {version}")
            group = [r for r in group if (path, clients, r["repeat"]) not in interrupted]
            if len(group) < 2:
                raise ValueError(f"Need at least two valid pairs for {path}, {clients}")
            counts = sum((Counter(r["status"]) for r in group), Counter())
            rates = median(r["success_rps"] for r in group)
            latency = [median(r["latency_ms"][k] for r in group) for k in ("p50", "p95", "p99")]
            other = sum(counts.values()) - counts["200"] - counts["429"]
            print(f"| {path} | {clients} | {version} | {rates:,.0f} | "
                  f"{latency[0]:.3f} | {latency[1]:.3f} | {latency[2]:.3f} | "
                  f"{counts['200']:,} | {counts['429']:,} | {other:,} |")


if __name__ == "__main__":
    main()
