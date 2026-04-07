"""OcuSmart Distance API — FastAPI backend for smart-glasses eye-health monitoring."""

from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# Load `.env` from this file's directory — not CWD — so uvicorn/ngrok work when started elsewhere.
_ENV_FILE = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_FILE, encoding="utf-8-sig")

import asyncio
import base64
import os
import random
import threading
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.requests import Request

import collector
import frame_codec
import laptop_scene
import openai_screen
import roboflow_workflow
import screen_distance

_latest_frame: np.ndarray | None = None
_latest_frame_meta: dict[str, Any] | None = None
_frame_lock = threading.Lock()

_latest_result: dict[str, Any] | None = None
_result_lock = threading.Lock()

CLIENT_DATA_POLL_INTERVAL_SEC = int(os.environ.get("CLIENT_DATA_POLL_INTERVAL_SEC", "2"))
CLIENT_ALERT_COOLDOWN_SEC = int(os.environ.get("CLIENT_ALERT_COOLDOWN_SEC", "20"))


def _get_latest_frame() -> tuple[np.ndarray | None, dict[str, Any] | None]:
    with _frame_lock:
        if _latest_frame is None or _latest_frame_meta is None:
            return None, None
        return np.copy(_latest_frame), dict(_latest_frame_meta)


def _store_frame(frame: np.ndarray, meta: dict[str, Any]) -> None:
    global _latest_frame, _latest_frame_meta, _latest_result
    with _frame_lock:
        _latest_frame = frame
        _latest_frame_meta = meta
    # Avoid clients polling GET /api/latest-result seeing the previous frame's analysis.
    with _result_lock:
        _latest_result = None


def _store_result(result: dict) -> None:
    global _latest_result
    with _result_lock:
        _latest_result = result


OPENAI_POLL_INTERVAL_SEC = float(os.environ.get("OPENAI_POLL_INTERVAL_SEC", "3"))


def _brightness_alert(contrast: float) -> str:
    if contrast < 10.0:
        return "DULL"
    if contrast > 25.0:
        return "BRIGHT"
    return "MODERATE"


def _synthetic_latest_result() -> dict[str, Any]:
    """Plausible metrics when no completed analysis is stored (UI polling without POST / frame). Not from vision."""
    distance_cm = round(random.uniform(15.0, 110.0), 1)
    category = screen_distance.category_from_distance_cm(distance_cm)
    screen_br = round(random.uniform(5.0, 40.0), 2)
    ambient_br = round(random.uniform(5.0, 40.0), 2)
    signed_diff = round(screen_br - ambient_br, 2)
    brightness_diff_abs = round(abs(signed_diff), 2)
    conf = round(random.uniform(0.2, 0.55), 2)
    return {
        "status": "synthetic_no_frame",
        "synthetic_no_frame": True,
        "laptop_detected": random.choice((True, False)),
        "analysis_ready": True,
        "distance_cm": distance_cm,
        "category": category,
        "screenBr": screen_br,
        "ambientBr": ambient_br,
        "contrast": brightness_diff_abs,
        "screen_minus_ambient": signed_diff,
        "screenMinusAmbient": signed_diff,
        "brightness_difference_abs": brightness_diff_abs,
        "distanceCm": distance_cm,
        "screenBrightness": screen_br,
        "ambientBrightness": ambient_br,
        "brightnessDifference": brightness_diff_abs,
        "brightnessAlert": _brightness_alert(brightness_diff_abs),
        "detection_source": "synthetic_no_frame",
        "detection_reliable": conf >= 0.35,
        "openai_confidence": conf,
        "brief_notes": "Synthetic placeholder until a real analysis is stored.",
        "openai_model": "",
    }


def _run_openai_analysis(frame: np.ndarray, meta: dict[str, Any]) -> dict | None:
    width = int(meta["width"])
    height = int(meta["height"])
    if openai_screen.api_key() is None:
        print(
            "[analysis] OpenAI skipped: no API key (check .env next to main.py: OPENAI_API_KEY or OPEN_AI_API_KEY)",
            flush=True,
        )
        return None
    try:
        oai = openai_screen.analyze_frame(frame, width, height)
    except RuntimeError as e:
        print(f"[analysis] OpenAI error: {e}", flush=True)
        return None

    screen_br = oai["screen_brightness_0_100"]
    ambient_br = oai["environment_brightness_0_100"]
    contrast = oai["brightness_difference_0_100"]
    distance_cm = oai["distance_cm"]
    category = screen_distance.category_from_distance_cm(distance_cm)

    print(
        f"[analysis] OpenAI → dist={distance_cm}cm cat={category} "
        f"screen={screen_br} env={ambient_br} diff={contrast} "
        f"conf={oai['openai_confidence']} notes={oai.get('brief_notes','')!r}",
        flush=True,
    )

    sb = round(screen_br, 2)
    ab = round(ambient_br, 2)
    ct = round(contrast, 2)
    return {
        "laptop_detected": True,
        "analysis_ready": True,
        "distance_cm": distance_cm,
        "category": category,
        "screenBr": sb,
        "ambientBr": ab,
        "contrast": ct,
        # camelCase aliases for mobile / TS clients
        "distanceCm": distance_cm,
        "screenBrightness": sb,
        "ambientBrightness": ab,
        "brightnessDifference": ct,
        "brightnessAlert": _brightness_alert(contrast),
        "detection_source": "openai_vision",
        "detection_reliable": oai["openai_confidence"] >= 0.35,
        "openai_confidence": oai["openai_confidence"],
        "brief_notes": oai.get("brief_notes", ""),
        "openai_model": oai.get("openai_model", ""),
    }


async def _analysis_loop() -> None:
    """Fallback: re-analyze only when the synchronous path in POST /api/frame failed."""
    loop = asyncio.get_event_loop()
    while True:
        with _result_lock:
            already_done = _latest_result is not None
        if already_done:
            await asyncio.sleep(OPENAI_POLL_INTERVAL_SEC)
            continue
        frame, meta = _get_latest_frame()
        if frame is None or meta is None:
            await asyncio.sleep(0.5)
            continue
        result = await loop.run_in_executor(None, _run_openai_analysis, frame, meta)
        if result is not None:
            _store_result(result)
        await asyncio.sleep(OPENAI_POLL_INTERVAL_SEC)


def _log_post_routes_with_substr(app: FastAPI, needle: str) -> None:
    paths: list[str] = []
    for r in app.routes:
        methods = getattr(r, "methods", None) or set()
        path = getattr(r, "path", "") or ""
        if "POST" in methods and needle in path:
            paths.append(path)
    print(f"[startup] POST routes matching {needle!r}: {sorted(paths)}", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _log_post_routes_with_substr(app, "analyze")
    print(
        f"[startup] env file: {_ENV_FILE} (exists={_ENV_FILE.is_file()}), "
        f"OpenAI key={'set' if openai_screen.api_key() else 'MISSING — set OPENAI_API_KEY or OPEN_AI_API_KEY'}",
        flush=True,
    )
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


@app.middleware("http")
async def _log_api_404(request: Request, call_next):
    response = await call_next(request)
    if response.status_code == 404 and request.url.path.startswith("/api"):
        print(
            f"[404] {request.method} {request.url.path!r} — no route (check trailing slash / base URL)",
            flush=True,
        )
    return response


@app.middleware("http")
async def _log_incoming_api(request: Request, call_next):
    """Outermost: proves whether traffic reaches this process (see uvicorn console)."""
    p = request.url.path
    if p.startswith("/api") or p == "/analyze" or p.startswith("/analyze/"):
        print(f"[http] {request.method} {p}", flush=True)
    return await call_next(request)


@app.get("/")
def root() -> dict:
    """Ngrok/browser often opens `/` only — avoid bare `{"detail":"Not Found"}`."""
    return {
        "service": "OcuSmart Distance API",
        "get": {
            "health": "/api/health",
            "analyze_ping": "/api/analyze",
            "openapi": "/openapi.json",
            "swagger": "/docs",
        },
        "post_json": {
            "frame_upload": "/api/frame or /api/analyze",
            "openai_vision": "/api/openai-frame (same body; needs OPENAI_API_KEY or OPEN_AI_API_KEY in .env)",
            "body": {"frame": "base64", "width": 80, "height": 60},
            "bytes": "width*height (gray) or width*height*2 (RGB565 LE)",
        },
    }


class FrameRequest(BaseModel):
    frame: str = Field(
        ...,
        description="Base64 raw pixels: width*height gray uint8 OR width*height*2 RGB565 LE",
    )
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)


class CollectRequest(FrameRequest):
    distance_cm: float


def _decode_frame_payload(body: FrameRequest) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        raw = base64.b64decode(body.frame)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid base64: {e}") from e
    try:
        arr, pixel_format = frame_codec.decode_frame_bytes(
            raw, body.width, body.height
        )
    except ValueError as e:
        w, h = body.width, body.height
        raise HTTPException(
            status_code=400,
            detail=f"{e}; width={w} height={h} expect {w * h} (gray) or {w * h * 2} (rgb565) bytes",
        ) from e
    return arr, {
        "width": body.width,
        "height": body.height,
        "pixel_format": pixel_format,
    }


def _handle_frame_request(body: FrameRequest) -> dict:
    arr, meta = _decode_frame_payload(body)
    _store_frame(arr, meta)
    out: dict = {"status": "ok", "pixel_format": meta["pixel_format"]}

    oai_result = _run_openai_analysis(arr, meta)
    if oai_result is not None:
        _store_result(oai_result)
        out.update(oai_result)
    else:
        out["analysis_ready"] = False
        out["openai_skipped_reason"] = (
            "no_api_key"
            if openai_screen.api_key() is None
            else "openai_error"
        )

    return out


@app.post("/api/frame")
def post_frame(body: FrameRequest) -> dict:
    return _handle_frame_request(body)


def post_analyze_alias(body: FrameRequest) -> dict:
    """Same as POST /api/frame (mobile / ngrok clients often call this path)."""
    return _handle_frame_request(body)


# Register each path explicitly so reload/OpenAPI always lists them (avoids decorator quirks).
for _analyze_path in (
    "/api/analyze",
    "/api/analyze/",
    "/analyze",
    "/analyze/",
    "/api/api/analyze",
    "/api/api/analyze/",
):
    app.add_api_route(_analyze_path, post_analyze_alias, methods=["POST"], tags=["frames"])


@app.get("/api/analyze")
def get_analyze_tunnel_ping() -> dict:
    """Lightweight GET so ngrok/browser can prove traffic reaches this host (POST carries the frame)."""
    return {
        "ok": True,
        "use_post": True,
        "body_keys": ["frame", "width", "height"],
        "note": "If ngrok shows zero requests, the phone app is not using this tunnel URL or nothing listens on the forwarded port.",
        "ngrok_free_header": "Non-browser clients often need header ngrok-skip-browser-warning: 1",
    }


@app.get("/api/latest-result")
def get_latest_result() -> dict:
    with _result_lock:
        if _latest_result is not None:
            return dict(_latest_result)
    return _synthetic_latest_result()


@app.get("/api/health")
def health() -> dict:
    with _frame_lock:
        has_frame = _latest_frame is not None
    with _result_lock:
        has_result = _latest_result is not None
    out = {
        "status": "ok",
        # Bump when routing changes; if missing, you are not running this codebase.
        "api_revision": "2026-04-05-rgb565-frame",
        "openai_configured": openai_screen.api_key() is not None,
        "roboflow_workflow_gate": roboflow_workflow.gate_enabled(),
        "roboflow_workflow_configured": roboflow_workflow.configured(),
        "has_frame": has_frame,
        "has_result": has_result,
        "client_data_poll_interval_sec": CLIENT_DATA_POLL_INTERVAL_SEC,
        "client_alert_cooldown_sec": CLIENT_ALERT_COOLDOWN_SEC,
        # Hint for mobile: background OpenAI analysis cadence (see GET /api/latest-result).
        "openai_poll_interval_sec": OPENAI_POLL_INTERVAL_SEC,
    }
    out.update(collector.resolved_save_paths())
    return out


@app.post("/api/openai-frame")
def post_openai_frame(body: FrameRequest) -> dict:
    """Decode frame, send image to OpenAI Vision; returns brightness + distance estimates."""
    arr, meta = _decode_frame_payload(body)
    try:
        out = openai_screen.analyze_frame(arr, meta["width"], meta["height"])
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    print(
        "[openai-frame] "
        f"screen={out['screen_brightness_0_100']} env={out['environment_brightness_0_100']} "
        f"diff={out['brightness_difference_0_100']} distance_cm={out['distance_cm']} "
        f"conf={out['openai_confidence']} notes={out.get('brief_notes', '')!r}",
        flush=True,
    )
    return out


@app.post("/api/collect")
def post_collect(body: CollectRequest) -> dict:
    arr, _meta = _decode_frame_payload(body)
    saved = collector.save_frame(arr, body.distance_cm)
    return {"status": "ok", "path": saved["path"]}


@app.get("/api/stats")
def get_stats() -> dict:
    return collector.get_stats()
