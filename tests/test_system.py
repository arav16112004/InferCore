"""Real-process tests. Build `cargo build --release` before running pytest."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import httpx
import numpy as np
import onnxruntime as ort
import pytest
from worker.model import MODEL, generate, train

ROOT = Path(__file__).resolve().parents[1]
VALUES = [6.7, 3.0, 5.2, 2.3]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for(check, timeout=10):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        try:
            if check():
                return
        except (httpx.HTTPError, KeyError):
            pass
        time.sleep(0.02)
    raise AssertionError("condition did not become true")


def stop(process):
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


@contextmanager
def gateway(urls, policy="round-robin", active=6, queue=12):
    port = free_port()
    env = dict(os.environ, BACKEND="onnx", WORKER_URLS=",".join(urls),
               POLICY=policy, BIND=f"127.0.0.1:{port}", MAX_IN_FLIGHT=str(active),
               QUEUE_CAPACITY=str(queue))
    p = subprocess.Popen([str(ROOT / "target/release/infercore")], env=env,
                         cwd=ROOT, stdout=subprocess.DEVNULL)
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", trust_env=False, timeout=3) as c:
            wait_for(lambda: c.get("/health").status_code == 200)
            yield c
    finally:
        stop(p)


@pytest.fixture
def workers():
    ports = [free_port() for _ in range(3)]
    processes = []
    def start(port):
        p = subprocess.Popen([sys.executable, "-m", "uvicorn", "worker.server:app",
            "--port", str(port), "--log-level", "error", "--no-access-log"], cwd=ROOT)
        processes.append(p)
        return p
    try:
        for port in ports:
            start(port)
        urls = [f"http://127.0.0.1:{p}" for p in ports]
        with httpx.Client(trust_env=False) as c:
            for url in urls:
                wait_for(lambda: c.get(url + "/health").status_code == 200, timeout=20)
        yield urls, processes, start, ports
    finally:
        for p in processes:
            stop(p)


def test_model_reproducible_and_matches_sklearn(tmp_path):
    path = tmp_path / "copy.onnx"
    metadata = generate(path)
    assert path.read_bytes() == MODEL.read_bytes()
    assert metadata["sha256"] == json.loads(MODEL.with_suffix(".json").read_text())["sha256"]
    clf, xs, _ = train()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    for x in xs:
        actual = session.run(None, {"features": x.reshape(1,4)})[0]
        np.testing.assert_allclose(actual, clf.predict_proba(x.reshape(1,4)), atol=2e-6)


@pytest.mark.parametrize("policy", ["round-robin", "least-in-flight"])
def test_three_workers_real_inference_concurrency_and_crash_recovery(workers, policy):
    urls, processes, start, ports = workers
    with gateway(urls, policy) as c:
        wait_for(lambda: c.get("/ready").status_code == 200)
        seen = set()
        for _ in range(9):
            r = c.post("/infer", json={"values": VALUES})
            assert r.status_code == 200
            body = r.json()
            assert body["backend"] == "onnx"
            for header in ("x-queue-wait-us", "x-worker-queue-us", "x-onnx-run-us"):
                assert int(r.headers[header]) >= 0
            assert len(body["output"]) == 2
            assert body["request_id"] == r.headers["x-request-id"]
            seen.add(body["worker_id"])
        assert seen == {0,1,2}
        # Kill a process while concurrent requests are being issued. Some requests
        # may lose both attempts; those must be explicit 503/504, never a hang.
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(c.post,"/infer",json={"values":VALUES}) for _ in range(80)]
            processes[0].kill()
            processes[0].wait()
            results = [f.result() for f in futures]
        assert all(r.status_code in (200,429,503,504) for r in results)
        assert any(r.status_code == 200 for r in results)
        wait_for(lambda: not c.get("/ready").json()["workers"][0]["healthy"])
        for _ in range(6):
            r=c.post("/infer",json={"values":VALUES})
            assert r.status_code == 200
            assert r.json()["worker_id"] != 0
        restarted=start(ports[0])
        assert restarted.pid != processes[0].pid
        wait_for(lambda: c.get("/ready").json()["workers"][0]["healthy"], timeout=20)
        seen={c.post("/infer",json={"values":VALUES}).json()["worker_id"] for _ in range(12)}
        assert 0 in seen
        wait_for(lambda: all(w["in_flight"] == 0 for w in c.get("/ready").json()["workers"]))


@contextmanager
def fake_worker(mode):
    """Controlled standard-HTTP fault peer; never used for real benchmarks."""
    class Handler(BaseHTTPRequestHandler):
        calls = 0
        def log_message(self,*args):
            pass
        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(b'{}')
        def do_POST(self):
            type(self).calls += 1
            self.rfile.read(int(self.headers["Content-Length"]))
            if mode in ("busy", "expired"):
                self.send_response(429 if mode == "busy" else 504)
                self.end_headers()
                return
            if mode == "disconnect":
                self.close_connection = True
                return
            if mode == "stall":
                time.sleep(0.3)
            self.send_response(200)
            self.send_header("Content-Type","application/json")
            self.end_headers()
            try:
                bodies = {
                    "malformed": b'{"output":[0.2]}',
                    "extra_score": b'{"output":[0.2,0.8,0.0]}',
                }
                self.wfile.write(bodies.get(mode, b'{"output":[0.2,0.8]}'))
            except BrokenPipeError:
                pass
    server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", Handler
    finally:
        server.shutdown(); server.server_close(); thread.join()


@pytest.mark.parametrize("mode", ["disconnect","stall","malformed","extra_score"])
def test_retry_different_worker_inside_original_deadline(mode):
    with fake_worker(mode) as (bad,calls), fake_worker("ok") as (good,good_calls):
        with gateway([bad,good],active=1,queue=0) as c:
            t=time.monotonic()
            r=c.post("/infer",json={"values":VALUES,"deadline_ms":120})
            elapsed=time.monotonic()-t
            assert r.status_code == 200
            assert r.json()["worker_id"] == 1
            assert calls.calls == 1 and good_calls.calls == 1
            assert elapsed < 0.25
            assert all(w["in_flight"] == 0 for w in c.get("/ready").json()["workers"])
            assert c.post("/infer",json={"values":VALUES}).status_code == 200


def test_all_workers_stall_deadline_and_capacity_recovery():
    with fake_worker("stall") as (a,_), fake_worker("stall") as (b,_):
        with gateway([a,b],active=1,queue=1) as c:
            t=time.monotonic()
            r=c.post("/infer",json={"values":VALUES,"deadline_ms":40})
            assert r.status_code == 504
            assert time.monotonic()-t < 0.2
            wait_for(lambda: all(w["in_flight"] == 0 for w in c.get("/ready").json()["workers"]))
            # Longer subsequent request succeeds: active capacity was released.
            wait_for(lambda: c.get("/ready").status_code == 200)
            assert c.post("/infer",json={"values":VALUES,"deadline_ms":1000}).status_code == 200


def test_worker_cancellation_keeps_compute_slot_until_native_work_finishes():
    from worker.server import app, lifespan
    async def scenario():
        entered=threading.Event()
        release=threading.Event()
        class BlockingSession:
            def run(self,*args):
                entered.set()
                assert release.wait(timeout=3)
                return [np.array([[.25,.75]],dtype=np.float32)]
        async with lifespan(app):
            app.state.session=BlockingSession()
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://test") as c:
                task=asyncio.create_task(c.post("/infer",json={"values":VALUES,"request_id":"test","remaining_ms":1000}))
                for _ in range(100):
                    if entered.is_set():
                        break
                    await asyncio.sleep(.005)
                assert entered.is_set()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                health = await c.get("/health")
                assert health.status_code == 200
                assert health.json()["pending"] == 1
                # A second job can wait, but a third must be rejected.
                queued=asyncio.create_task(c.post("/infer",json={"values":VALUES,"request_id":"queued","remaining_ms":1000}))
                await asyncio.sleep(.01)
                assert (await c.post("/infer",json={"values":VALUES,"request_id":"overload","remaining_ms":1000})).status_code == 429
                release.set()
                assert (await queued).status_code == 200
                assert (await c.get("/health")).json()["pending"] == 0
    asyncio.run(scenario())


def test_worker_skips_expired_queued_job_and_reuses_slot():
    from worker.server import app, lifespan

    async def scenario():
        entered, release = threading.Event(), threading.Event()
        executed = []

        class BlockingSession:
            def run(self, names, feeds):
                executed.append(feeds["features"][0].tolist())
                entered.set()
                if not release.wait(timeout=3):
                    raise RuntimeError("timed out waiting for test release")
                return [np.array([[0.25, 0.75]], dtype=np.float32)]

        async with lifespan(app):
            app.state.session = BlockingSession()
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                payload = {"values": VALUES, "request_id": "first", "remaining_ms": 1000}
                first = asyncio.create_task(client.post("/infer", json=payload))
                try:
                    assert await asyncio.to_thread(entered.wait, 2)
                    queued = asyncio.create_task(client.post("/infer", json={**payload, "request_id":"waiter", "remaining_ms":5}))
                    for _ in range(100):
                        if (await client.get("/health")).json()["pending"] == 2:
                            break
                        await asyncio.sleep(.001)
                    else:
                        pytest.fail("second request was never admitted")
                    await asyncio.sleep(.02)
                finally:
                    release.set()
                assert (await first).status_code == 200
                assert (await queued).status_code == 504
                assert len(executed) == 1
                assert (await client.post("/infer", json=payload)).status_code == 200
                assert (await client.get("/health")).json()["pending"] == 0

    asyncio.run(scenario())


@pytest.mark.parametrize(("mode", "status"), [("busy", 429), ("expired", 504)])
def test_worker_rejection_preserves_health_and_can_retry(mode, status):
    with fake_worker(mode) as (worker, calls):
        with gateway([worker]) as c:
            for _ in range(2):
                assert c.post("/infer", json={"values": VALUES}).status_code == status
                assert c.get("/ready").json()["workers"][0]["healthy"]
            assert calls.calls == 2
        with fake_worker("ok") as (available, alternate_calls):
            with gateway([worker, available]) as c:
                reply = c.post("/infer", json={"values": VALUES})
                assert reply.status_code == 200
                assert reply.json()["worker_id"] == 1
                assert c.get("/ready").json()["workers"][0]["healthy"]
                assert calls.calls == 3 and alternate_calls.calls == 1


def test_only_configured_worker_gets_entire_remaining_deadline():
    # The peer returns after 300 ms, beyond half the budget but within 500 ms.
    with fake_worker("stall") as (url, calls):
        with gateway([url]) as c:
            reply = c.post("/infer", json={"values": VALUES, "deadline_ms": 500})
            assert reply.status_code == 200
            assert calls.calls == 1
