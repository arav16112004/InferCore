"""Matched ORT / worker HTTP / gateway profiling. See docs/OPTIMIZATION.md."""
import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

import httpx
import numpy as np
import onnxruntime as ort

from scripts.benchmark import percentile

ROOT = Path(__file__).resolve().parents[1]
VALUES = [[6.7, 3.0, 5.2, 2.3], [5.7, 2.8, 4.1, 1.3], [6.1, 2.7, 5.0, 1.7],
          [7.1, 3.0, 5.9, 2.1], [5.5, 2.4, 3.7, 1.0], [6.2, 2.9, 4.3, 1.3]]
# Single-worker paths localize overhead; three-worker paths reflect default deployment.
SCENARIOS = [('worker', 1), ('gateway1', 1), ('worker', 12), ('gateway1', 12),
             ('round-robin', 12), ('least-in-flight', 12),
             ('round-robin', 48), ('least-in-flight', 48)]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(samples):
    return {f'p{p}': percentile(samples, p/100) for p in (50, 95, 99)}


def wait(url, child, predicate):
    end = time.monotonic() + 25
    with httpx.Client(trust_env=False, timeout=0.5) as client:
        while time.monotonic() < end:
            if child.poll() is not None:
                raise RuntimeError(f'Child {child.pid} exited before readiness: {url}')
            try:
                if predicate(client.get(url).raise_for_status().json()):
                    return
            except httpx.HTTPError:
                pass
            time.sleep(.025)
    raise TimeoutError(url)


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def stop(child):
    if child.poll() is None:
        child.send_signal(signal.SIGINT)
        try:
            child.wait(timeout=4)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def start_stack(source, stage, model, stack, native=False):
    # Both revisions serve identical model bytes. Use an isolated cwd and compare
    # health PIDs to the new processes, so a running demo cannot be mistaken for them.
    shutil.copytree(source/'worker', stage/'worker', ignore=shutil.ignore_patterns('__pycache__'))
    (stage/'models').mkdir()
    shutil.copy(model, stage/'models/iris_logistic.onnx')
    urls = []
    endpoints = {}
    ready_ms = []
    for i in range(3):
        port = free_port()
        start = time.perf_counter()
        child = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'worker.server:app',
            '--host', '127.0.0.1', '--port', str(port), '--loop', 'uvloop' if native else 'asyncio',
            '--http', 'httptools' if native else 'h11',
            '--log-level', 'error', '--no-access-log'], cwd=stage)
        stack.callback(stop, child)
        url = f'http://127.0.0.1:{port}'
        wait(url+'/health', child, lambda data: data.get('pid') == child.pid)
        ready_ms.append((time.perf_counter()-start)*1000)
        urls.append(url)
        if i == 0:
            endpoints['worker'] = port
    for path, n, policy in [('gateway1', 1, 'round-robin'), ('round-robin', 3, 'round-robin'),
                            ('least-in-flight', 3, 'least-in-flight')]:
        port = free_port()
        env = dict(os.environ, BACKEND='onnx', BIND=f'127.0.0.1:{port}', POLICY=policy,
                   MAX_IN_FLIGHT='6', QUEUE_CAPACITY='12', WORKER_URLS=','.join(urls[:n]))
        proc = subprocess.Popen([str(source/'target/release/infercore')], env=env, stdout=subprocess.DEVNULL)
        stack.callback(stop, proc)
        wait(f'http://127.0.0.1:{port}/ready', proc,
             lambda data: [w['url'] for w in data['workers'] if w['healthy']] == urls[:n])
        endpoints[path] = port
    return endpoints, ready_ms


def direct_ort(model, output):
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model), options, providers=['CPUExecutionProvider'])
    inputs = [{'features': np.asarray([v], dtype=np.float32)} for v in VALUES]
    cases = [dict(values=values, expected=session.run(None, data)[0][0].tolist())
             for values, data in zip(VALUES, inputs)]
    for i in range(500):
        session.run(None, inputs[i % len(inputs)])
    results = []
    with gzip.open(output/'direct-ort-us.jsonl.gz', 'wt') as f:
        for _ in range(3):
            times = []
            for i in range(10000):
                data = inputs[i % len(inputs)]
                begin = time.perf_counter_ns()
                session.run(None, data)
                times.append((time.perf_counter_ns()-begin)/1000)
            f.write(json.dumps(times)+'\n')
            results.append(stats(times))
    return cases, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True, help='old checkout with target/release/infercore')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=3.0)
    parser.add_argument('--warmup', type=float, default=1.0)
    parser.add_argument('--repeats', type=int, default=4)
    parser.add_argument('--native', action='store_true', help='use uvloop + httptools for optimized workers')
    args = parser.parse_args()
    if min(args.warmup, args.seconds, args.repeats) <= 0:
        parser.error('positive warmup, measurement duration and repeats required')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    model = ROOT/'models/iris_logistic.onnx'
    sources = {'baseline': args.baseline.resolve(), 'optimized': ROOT}
    cases, ort_times = direct_ort(model, output)
    run = {'timestamp': datetime.now(timezone.utc).isoformat(), 'platform': platform.platform(),
           'cpu': subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),
           'logical_cpus': os.cpu_count(), 'ram_bytes': int(subprocess.check_output(['sysctl','-n','hw.memsize'])),
           'python': platform.python_version(),
           'rust': subprocess.check_output(['rustc', '-V'], text=True).strip(),
           'driver_sha256': digest(Path(__file__).resolve()),
           'client_sha256': digest(ROOT/'scripts/profile_client.mjs'),
           'rust_build': 'cargo build --release --locked',
           'clock': 'monotonic wall time including drain', 'load': 'closed-loop, one keep-alive connection per caller',
           'node': subprocess.check_output(['node','-v'],text=True).strip(),
           'ort': ort.__version__, 'model_sha256': digest(model), 'workload': cases,
           'workers': 3, 'per_worker_depth': 2, 'per_worker_compute': 1, 'active': 6, 'queue': 12,
           'deadline_ms': 1000, 'warmup_s': args.warmup, 'measurement_s': args.seconds,
           'repeats': args.repeats, 'ort_us': ort_times,
           'baseline_worker_http': 'h11/asyncio',
           'optimized_worker_http': 'httptools/uvloop' if args.native else 'h11/asyncio',
           'rows': ['relative_start_ms','latency_ms','status','admission_us','worker_id','error','worker_queue_us','ort_run_us'],
           'packages': subprocess.check_output([sys.executable,'-m','pip','freeze'], text=True).splitlines(),
           'files': {v: {str(p.relative_to(s)): digest(p) for p in
                     [*sorted((s/'worker').glob('*.py')), *sorted((s/'src').glob('*.rs')), s/'target/release/infercore',s/'Cargo.lock']}
                     for v,s in sources.items()}}
    summaries = []
    with tempfile.TemporaryDirectory() as staging, ExitStack() as stack:
        endpoints = {}
        run['worker_ready_ms'] = {}
        for label, source in sources.items():
            stage = Path(staging)/label
            endpoints[label], run['worker_ready_ms'][label] = start_stack(
                source, stage, model, stack, native=(label == 'optimized' and args.native))
        (output/'manifest.json').write_text(json.dumps(run, indent=2)+'\n')
        for path, concurrency in SCENARIOS:
            for i in range(args.repeats):
                order = list(sources) if i%2 == 0 else list(reversed(sources))
                for label in order:
                    settings = dict(port=endpoints[label][path], concurrency=concurrency, direct=path=='worker',
                                    cases=cases, warmup=args.warmup, duration=args.seconds)
                    result = subprocess.run(['node', ROOT/'scripts/profile_client.mjs'], input=json.dumps(settings),
                                            capture_output=True, text=True, check=True)
                    data = json.loads(result.stdout)
                    rows = data['rows']
                    key = f'{path}-c{concurrency}-{label}-{i+1}'
                    with gzip.open(output/(key+'.json.gz'), 'wt') as f:
                        json.dump(rows, f, separators=(',',':'))
                    ok = [row for row in rows if row[2] == 200]
                    queue = [r[3] for r in ok if r[3] is not None]
                    summary = dict(key=key, path=path, concurrency=concurrency, variant=label,
                                   repeat=i+1, seconds=data['elapsed_s'], status=Counter(r[2] for r in rows),
                                   success_rps=len(ok)/data['elapsed_s'], latency_ms=stats([r[1] for r in ok]),
                                   admission_us=stats(queue), num_queue_measurements=len(queue),
                                   worker_queue_us=stats([r[6] for r in ok if r[6] is not None]),
                                   ort_run_us=stats([r[7] for r in ok if r[7] is not None]))
                    summaries.append(summary)
                    (output/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
                    print(key, round(summary['success_rps']), summary['status'], flush=True)


if __name__ == '__main__':
    main()
