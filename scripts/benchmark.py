"""Closed-loop HTTP benchmark. Raw per-request records accompany every summary."""
import argparse
import asyncio
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
import httpx

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.local import start_worker, wait_ready


def command(args):
    return subprocess.check_output(args,cwd=ROOT,text=True).strip()


def percentile(values,p):
    if not values:
        return None
    s=sorted(values)
    return s[max(0,math.ceil(len(s)*p)-1)]


def distribution(values):
    buckets=[1,2,5,10,20,50,100,250,500,1000,10000]
    return {"count":len(values), "p50_ms":percentile(values,.5),
        "p95_ms":percentile(values,.95),"p99_ms":percentile(values,.99),
        "histogram_cumulative_ms": {str(b):sum(v<=b for v in values) for b in buckets} | {"+Inf":len(values)}}


async def phase(client, concurrency, seconds):
    rows=[]
    started=time.perf_counter()
    until=started+seconds
    async def user(user_id):
        while time.perf_counter()<until:
            begin=time.perf_counter()
            status=0; error=None; worker=None
            try:
                r=await client.post("/infer",json={"values":[6.7,3.0,5.2,2.3],"deadline_ms":1000})
                status=r.status_code
                if status==200:
                    body=r.json()
                    if body.get("backend")!="onnx" or len(body.get("output",[]))!=2:
                        error="invalid_success_body"; status=0
                    worker=body.get("worker_id")
            except (httpx.HTTPError,ValueError) as e:
                error=type(e).__name__
            rows.append({"user":user_id,"start_s":begin-started,
                "latency_ms":(time.perf_counter()-begin)*1000,
                "status":status,"error":error,"worker_id":worker})
    await asyncio.gather(*(user(i) for i in range(concurrency)))
    return rows,time.perf_counter()-started


async def measure(concurrency,warmup,seconds):
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8080",trust_env=False,
            timeout=12,limits=httpx.Limits(max_connections=concurrency,max_keepalive_connections=concurrency)) as c:
        warm,_=await phase(c,concurrency,warmup)
        rows,elapsed=await phase(c,concurrency,seconds)
    success=[r["latency_ms"] for r in rows if r["status"]==200]
    counts=Counter(str(r["status"]) for r in rows)
    return rows,{"elapsed_including_drain_s":elapsed,"warmup_requests":len(warm),
        "requests":len(rows),"status_counts":dict(counts),
        "success_rps":len(success)/elapsed,"total_rps":len(rows)/elapsed,
        "overload_rejections":counts["429"],"errors_excluding_overload":sum(v for k,v in counts.items() if k not in ("200","429")),
        "successful_latency":distribution(success),
        "latency_by_status":{k:distribution([r["latency_ms"] for r in rows if str(r["status"])==k]) for k in counts},
        "worker_counts":dict(Counter(str(r["worker_id"]) for r in rows if r["status"]==200))}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seconds",type=float,default=5)
    p.add_argument("--warmup",type=float,default=1)
    p.add_argument("--repeats",type=int,default=3)
    p.add_argument("--concurrency",type=int,nargs="+",default=[1,12,48])
    p.add_argument("--output",type=Path,default=ROOT/"benchmarks/local")
    args=p.parse_args()
    if args.seconds<=0 or args.warmup<=0 or args.repeats<1 or min(args.concurrency)<1:
        p.error("durations, repeats, concurrency must be positive")
    if args.output.exists():
        p.error("output already exists; choose a new directory to preserve raw evidence")
    args.output.mkdir(parents=True)
    manifest={"timestamp_utc":datetime.now(timezone.utc).isoformat(),
        "platform":platform.platform(),"architecture":platform.machine(),
        "cpu":command(["sysctl","-n","machdep.cpu.brand_string"]),
        "memory_bytes":int(command(["sysctl","-n","hw.memsize"])),
        "logical_cpus":int(command(["sysctl","-n","hw.ncpu"])),
        "python":sys.version,"rust":command([str(ROOT/".tools/cargo/bin/rustc"),"-Vv"]),
        "python_dependencies":command([sys.executable,"-m","pip","freeze"]).splitlines(),
        "git_commit":command(["git","rev-parse","HEAD"]),
        "git_status":command(["git","status","--short"]),
        "binary_sha256":hashlib.sha256((ROOT/"target/release/infercore").read_bytes()).hexdigest(),
        "source_sha256":{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
            for folder in ["src","worker","scripts"] for path in sorted((ROOT/folder).glob("*")) if path.is_file()},
        "model":json.loads((ROOT/"models/iris_logistic.json").read_text()),
        "workers":3,"worker_compute_slots":1,"worker_pending_limit":2,
        "ort_intra_op_threads":1,"ort_inter_op_threads":1,
        "max_in_flight":6,"queue_capacity":12,"build":"cargo build --release --locked",
        "concurrency":args.concurrency,"warmup_seconds":args.warmup,
        "measurement_seconds":args.seconds,"repeats":args.repeats,
        "workload":{"values":[6.7,3.0,5.2,2.3],"deadline_ms":1000},
        "load_generator":"HTTPX async closed-loop on same host; no think time; keepalive; no proxy",
        "percentile_method":"nearest rank, successful responses only; failures separately",
        "caveats":["includes loopback HTTP, serialization and client overhead",
                   "tiny model; not representative of expensive AI inference",
                   "same-host client competes for CPU; closed-loop load omits offered-rate stalls",
                   "5-second default samples are exploratory, not production capacity certification"]}
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    workers=[]; gateway=None; summaries=[]
    try:
        for port in (9001,9002,9003):
            workers.append(start_worker(port))
        for port in (9001,9002,9003):
            wait_ready(f"http://127.0.0.1:{port}/health",timeout=30)
        for concurrency in args.concurrency:
            for repeat in range(args.repeats):
                policies=["round-robin","least-in-flight"]
                if repeat%2:
                    policies.reverse()
                for policy in policies:
                    run=f"c{concurrency}-r{repeat+1}-{policy}"
                    env=dict(os.environ,BACKEND="onnx",POLICY=policy,MAX_IN_FLIGHT="6",QUEUE_CAPACITY="12",
                        BIND="127.0.0.1:8080",WORKER_URLS="http://127.0.0.1:9001,http://127.0.0.1:9002,http://127.0.0.1:9003")
                    gateway=subprocess.Popen([str(ROOT/"target/release/infercore")],env=env,cwd=ROOT,stdout=subprocess.DEVNULL)
                    wait_ready("http://127.0.0.1:8080/ready")
                    rows,summary=asyncio.run(measure(concurrency,args.warmup,args.seconds))
                    summary.update(run=run,policy=policy,concurrency=concurrency,repeat=repeat+1)
                    with (args.output/f"{run}.jsonl").open("w") as f:
                        for row in rows:
                            f.write(json.dumps(row,separators=(",",":"))+"\n")
                    summaries.append(summary)
                    (args.output/"summary.json").write_text(json.dumps(summaries,indent=2)+"\n")
                    print(f'{run}: {summary["success_rps"]:.1f} success/s; p95={summary["successful_latency"]["p95_ms"]:.2f} ms; statuses={summary["status_counts"]}',flush=True)
                    gateway.send_signal(__import__("signal").SIGINT); gateway.wait(timeout=15); gateway=None
    finally:
        for proc in ([gateway] if gateway else [])+workers:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait()


if __name__=="__main__":
    main()
