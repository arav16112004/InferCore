"""A private HTTP worker with one CPU thread and at most two outstanding jobs."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import math
import os
import time

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_core import to_json

from .paths import MODEL


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")
    values: list[float] = Field(min_length=4, max_length=4)
    request_id: str = Field(min_length=1, max_length=64)
    remaining_ms: int = Field(ge=1, le=10000)

    @field_validator("values")
    @classmethod
    def finite(cls, values):
        if not all(math.isfinite(v) and abs(v) <= 1000 for v in values):
            raise ValueError("features must be finite and in [-1000,1000]")
        return values


@asynccontextmanager
async def lifespan(app):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(MODEL), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    session.run(None, {"features": np.zeros((1, 4), dtype=np.float32)})
    app.state.session = session
    app.state.jobs = set()
    # The executor's sole thread serializes model calls even after HTTP cancellation.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="infercore-ort") as executor:
        app.state.executor = executor
        try:
            yield
        finally:
            await asyncio.gather(*app.state.jobs, return_exceptions=True)


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "pid": os.getpid(), "pending": len(app.state.jobs),
            "backend": "onnxruntime-cpu"}


def compute(session, values, submitted_ns, deadline_ns):
    started_ns = time.monotonic_ns()
    if started_ns >= deadline_ns:
        raise HTTPException(504, "expired before computation")
    feed = {"features": np.asarray([values], dtype=np.float32)}
    before_ort_ns = time.monotonic_ns()
    result = session.run(["scores"], feed)
    ort_us = (time.monotonic_ns() - before_ort_ns) // 1000
    return to_json({"output": result[0][0].tolist()}), (started_ns - submitted_ns) // 1000, ort_us


def finished(job):
    app.state.jobs.discard(job)
    if not job.cancelled():
        job.exception()  # Observe errors even when the caller has already disconnected.


@app.post("/infer")
async def infer(request: Request):
    jobs = app.state.jobs
    if len(jobs) >= 2:
        raise HTTPException(429, "worker full")
    submitted_ns = time.monotonic_ns()
    deadline_ns = submitted_ns + request.remaining_ms * 1_000_000
    job = asyncio.get_running_loop().run_in_executor(
        app.state.executor, compute, app.state.session, request.values, submitted_ns, deadline_ns
    )
    # Admission and registration happen on the event loop with no intervening await.
    jobs.add(job)
    job.add_done_callback(finished)
    body, queue_us, ort_us = await asyncio.shield(job)
    return Response(content=body, media_type="application/json",
                    headers={"x-worker-queue-us": str(queue_us), "x-onnx-run-us": str(ort_us)})
