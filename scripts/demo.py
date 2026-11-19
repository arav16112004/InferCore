"""Reproducible terminal demo; writes an actual request/crash/recovery transcript."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time
import os
import signal
import httpx

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.local import wait_ready


def main():
    out=ROOT/"docs/demo.json"
    events=[]
    started=time.monotonic()
    def record(event,**data):
        row={"elapsed_s":round(time.monotonic()-started,3),"event":event,**data}
        events.append(row)
        print(json.dumps(row),flush=True)
    p=subprocess.Popen([sys.executable,"scripts/local.py"],cwd=ROOT)
    try:
        wait_ready("http://127.0.0.1:8080/ready",timeout=30)
        with httpx.Client(base_url="http://127.0.0.1:8080",trust_env=False,timeout=3) as c:
            def request(_):
                r=c.post("/infer",json={"values":[6.7,3.0,5.2,2.3]})
                return {"status":r.status_code,"id":r.headers.get("x-request-id"),"worker":r.json().get("worker_id")}
            record("ready",state=c.get("/ready").json())
            with ThreadPoolExecutor(max_workers=12) as executor:
                record("concurrent_requests",responses=list(executor.map(request,range(24))))
                old=httpx.get("http://127.0.0.1:9001/health",trust_env=False).json()["pid"]
                futures=[executor.submit(request,i) for i in range(80)]
                os.kill(old,signal.SIGKILL)
                record("worker_killed",pid=old,port=9001)
                responses=[f.result() for f in futures]
                record("requests_during_failure",responses=responses)
            last=None; new=None
            until=time.monotonic()+20
            while time.monotonic()<until:
                state=c.get("/ready").json()
                flags=[w["healthy"] for w in state["workers"]]
                if flags!=last:
                    record("health_transition",healthy=flags); last=flags
                try:
                    new=httpx.get("http://127.0.0.1:9001/health",timeout=.2,trust_env=False).json()["pid"]
                except httpx.HTTPError:
                    pass
                if new and new!=old and flags[0]:
                    break
                time.sleep(.05)
            if not new or new==old or not last[0]:
                raise RuntimeError("worker did not recover")
            responses=[request(i) for i in range(12)]
            assert any(r["worker"]==0 and r["status"]==200 for r in responses)
            record("recovered",old_pid=old,new_pid=new,responses=responses)
    finally:
        p.send_signal(signal.SIGINT)
        p.wait(timeout=20)
        out.write_text(json.dumps(events,indent=2)+"\n")


if __name__=="__main__":
    main()
