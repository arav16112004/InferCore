"""Start three independent workers and a gateway; restart crashed workers.

Uses standard HTTP, not a custom IPC protocol. This supervisor is deliberately
separate from Rust scheduling. Ctrl-C stops only processes owned by this launcher.
"""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import httpx

ROOT = Path(__file__).resolve().parents[1]


def wait_ready(url, timeout=15):
    until = time.monotonic() + timeout
    with httpx.Client(trust_env=False, timeout=0.5) as client:
        while time.monotonic() < until:
            try:
                if client.get(url).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.05)
    raise RuntimeError(f"not ready: {url}")


def start_worker(port):
    return subprocess.Popen([sys.executable, "-m", "uvicorn", "worker.server:app",
        "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning",
        "--no-access-log"], cwd=ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["round-robin", "least-in-flight"], default="round-robin")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("workers must be in [1,16]")
    processes = {}
    gateway = None
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        for port in range(9001, 9001 + args.workers):
            processes[port] = start_worker(port)
        for port in processes:
            wait_ready(f"http://127.0.0.1:{port}/health")
        env = dict(os.environ, BACKEND="onnx", POLICY=args.policy,
            WORKER_URLS=",".join(f"http://127.0.0.1:{p}" for p in processes))
        gateway = subprocess.Popen([str(ROOT / "target/release/infercore")], cwd=ROOT, env=env)
        wait_ready("http://127.0.0.1:8080/ready")
        print(f"Ready: gateway PID={gateway.pid}; workers=" + str({p: x.pid for p, x in processes.items()}), flush=True)
        while not stopping:
            if gateway.poll() is not None:
                raise RuntimeError(f"gateway exited {gateway.returncode}")
            for port, process in list(processes.items()):
                if process.poll() is not None:
                    print(f"Restarting worker port={port} exit={process.returncode}", flush=True)
                    processes[port] = start_worker(port)
            time.sleep(0.25)
    finally:
        children = ([gateway] if gateway else []) + list(processes.values())
        for process in children:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
        for process in children:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
