"""OcuSmart Distance API — FastAPI backend for smart-glasses eye-health monitoring."""

from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

import asyncio
import base64
import os
import threading
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import collector
import laptop_scene
import roboflow_workflow

_latest_frame: np.ndarray | None = None
_latest_frame_meta: dict[str, int] | None = None
_frame_lock = threading.Lock()

_latest_result: dict[str, Any] | None = None
_result_lock = threading.Lock()

CLIENT_DATA_POLL_INTERVAL_SEC = int(os.environ.get("CLIENT_DATA_POLL_INTERVAL_SEC", "2"))
CLIENT_ALERT_COOLDOWN_SEC = int(os.environ.get("CLIENT_ALERT_COOLDOWN_SEC", "20"))


def _get_latest_frame() -> tuple[np.ndarray | None, dict[str, int] | None]:
    with _frame_lock:
        if _latest_frame is None or _latest_frame_meta is None:
            return None, None
        return np.copy(_latest_frame), dict(_latest_frame_meta)


def _store_frame(frame: np.ndarray, meta: dict[str, int]) -> None:
    global _latest_frame, _latest_frame_meta
    with _frame_lock:
        _latest_frame = frame
        _latest_frame_meta = meta


def _store_result(result: dict) -> None:
    global _latest_result
    with _result_lock:
        _latest_result = result


def _run_analysis(frame: np.ndarray, meta: dict[str, int]) -> dict | None:
    width = int(meta["width"])
    height = int(meta["height"])
    det = roboflow_workflow.try_predict_for_gate(frame, width, height)
    if det is None:
        return None
    latest, latest_meta = _get_latest_frame()
    if latest is None or latest_meta is None:
        return None
    lw = int(latest_meta["width"])
    lh = int(latest_meta["height"])
    reliable = True
    return laptop_scene.analyze_with_laptop_gate(
        latest,
        lw,
        lh,
        cached_laptop_detection=(det, "roboflow", reliable),
    )


async def _analysis_loop() -> None:
    loop = asyncio.get_event_loop()
    while True:
        frame, meta = _get_latest_frame()
        if frame is None or meta is None:
            await asyncio.sleep(0.2)
            continue
        result = await loop.run_in_executor(None, _run_analysis, frame, meta)
        if result is not None:
            _store_result(result)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_analysis_loop())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="OcuSmart Distance API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class FrameRequest(BaseModel):
    frame: str = Field(..., description="Base64-encoded raw grayscale uint8 bytes")
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)


class CollectRequest(FrameRequest):
    distance_cm: float


def _decode_frame_payload(body: FrameRequest) -> tuple[np.ndarray, dict[str, int]]:
    try:
        raw = base64.b64decode(body.frame)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid base64: {e}") from e
    expected = body.width * body.height
    if len(raw) != expected:
        raise HTTPException(
            status_code=400,
            detail=f"decoded length {len(raw)} != width*height ({expected})",
        )
    arr = np.frombuffer(raw, dtype=np.uint8).copy()
    return arr, {"width": body.width, "height": body.height}


@app.post("/api/frame")
def post_frame(body: FrameRequest) -> dict:
    arr, meta = _decode_frame_payload(body)
    _store_frame(arr, meta)
    return {"status": "ok"}


@app.get("/api/latest-result")
def get_latest_result() -> dict:
    with _result_lock:
        if _latest_result is None:
            return {"status": "pending"}
        return dict(_latest_result)


@app.get("/api/health")
def health() -> dict:
    with _frame_lock:
        has_frame = _latest_frame is not None
    with _result_lock:
        has_result = _latest_result is not None
    return {
        "status": "ok",
        "roboflow_workflow_gate": roboflow_workflow.gate_enabled(),
        "roboflow_workflow_configured": roboflow_workflow.configured(),
        "has_frame": has_frame,
        "has_result": has_result,
        "client_data_poll_interval_sec": CLIENT_DATA_POLL_INTERVAL_SEC,
        "client_alert_cooldown_sec": CLIENT_ALERT_COOLDOWN_SEC,
    }


@app.post("/api/collect")
def post_collect(body: CollectRequest) -> dict:
    arr, _meta = _decode_frame_payload(body)
    saved = collector.save_frame(arr, body.distance_cm)
    return {"status": "ok", "path": saved["path"]}


@app.get("/api/stats")
def get_stats() -> dict:
    return collector.get_stats()
