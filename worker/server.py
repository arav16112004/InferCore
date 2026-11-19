"""One process, one ORT session, one CPU compute slot, at most two jobs.

The private HTTP interface is intended only for localhost/a private Docker network.
"""
import asyncio
import os
import time
from contextlib import asynccontextmanager
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator
from .model import MODEL


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")
    values: list[float] = Field(min_length=4, max_length=4)
    request_id: str = Field(min_length=1, max_length=64)
    remaining_ms: int = Field(ge=1, le=10000)

    @field_validator("values")
    @classmethod
    def finite(cls, values):
        if not all(np.isfinite(v) and abs(v) <= 1000 for v in values):
            raise ValueError("features must be finite and in [-1000,1000]")
        return values


@asynccontextmanager
async def lifespan(app):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    app.state.session = ort.InferenceSession(str(MODEL), sess_options=options,
                                            providers=["CPUExecutionProvider"])
    app.state.session.run(None, {"features": np.zeros((1, 4), dtype=np.float32)})
    app.state.compute = asyncio.Lock()
    app.state.pending = 0
    app.state.tasks = set()
    yield
    # Finish already admitted jobs before process shutdown.
    await asyncio.gather(*app.state.tasks, return_exceptions=True)


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "pid": os.getpid(), "pending": app.state.pending,
            "backend": "onnxruntime-cpu"}


async def execute(request, deadline):
    try:
        async with app.state.compute:
            if time.monotonic() >= deadline:
                raise HTTPException(504, "expired before computation")
            # to_thread keeps Python's event loop available for health requests.
            result = await asyncio.to_thread(app.state.session.run, ["scores"],
                {"features": np.asarray([request.values], dtype=np.float32)})
            return {"output": result[0][0].tolist()}
    finally:
        app.state.pending -= 1


@app.post("/infer")
async def infer(request: Request):
    if app.state.pending >= 2:
        raise HTTPException(429, "worker full")
    app.state.pending += 1  # no await between check and increment
    task = asyncio.create_task(execute(request, time.monotonic() + request.remaining_ms / 1000))
    app.state.tasks.add(task)
    def done(t):
        app.state.tasks.discard(t)
        if not t.cancelled():
            t.exception()  # retrieve errors even if the HTTP caller disconnected
    task.add_done_callback(done)
    # Cancelling the HTTP coroutine does not release the compute lock while ORT
    # continues in a native thread. The tracked task owns that lock until completion.
    return await asyncio.shield(task)
