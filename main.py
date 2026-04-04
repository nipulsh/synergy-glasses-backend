"""
OcuSmart Distance API

Endpoints
---------
GET  /api/health          → {"status": "ok", "model_loaded": bool}
POST /api/distance        → distance + brightness (ML if models/brightness_model.tflite exists)
POST /api/analyze         → distance + brightness; optional ?include_objects=1
POST /api/objects         → laptop/object classifier (objects_model.tflite)
POST /api/collect         → save labeled frame to training dataset
GET  /api/stats           → dataset frame counts per distance class

Frame format (POST body)
------------------------
{
  "frame":    "<base64-encoded raw grayscale bytes>",
  "width":    80,   // pixels (default matches ESP32 sub-sampled output)
  "height":   60
}
"""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import brightness_inference as bright_ml
import collector as col
import inference as inf
import laptop_scene as lap
import object_inference as obj_inf
import roboflow_workflow as rwf

app = FastAPI(
    title="OcuSmart Distance API",
    description="Receives camera frames from the phone app and returns ML-based distance estimates.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class FrameRequest(BaseModel):
    frame: str = Field(..., description="Base64-encoded raw grayscale bytes (width × height)")
    width: int = Field(80, ge=1, le=640)
    height: int = Field(60, ge=1, le=480)


class CollectRequest(FrameRequest):
    distance_cm: float = Field(..., ge=5, le=200, description="True distance in centimetres")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": inf._interp is not None,
        "model_path": str(inf.MODEL_PATH) if inf.MODEL_PATH.exists() else None,
        "objects_model_loaded": obj_inf.objects_model_loaded(),
        "objects_model_path": str(obj_inf.OBJECTS_MODEL_PATH)
        if obj_inf.OBJECTS_MODEL_PATH.exists()
        else None,
        "brightness_model_loaded": bright_ml.brightness_model_loaded(),
        "brightness_model_path": str(bright_ml.BRIGHTNESS_MODEL_PATH)
        if bright_ml.BRIGHTNESS_MODEL_PATH.exists()
        else None,
        "roboflow_workflow_gate": rwf.gate_enabled(),
        "roboflow_workflow_configured": rwf.configured(),
    }


def _gated_distance_body(frame: np.ndarray, width: int, height: int, include_laptop_details: bool):
    """If Roboflow workflow gate or objects_model is active: laptop-only pipeline; else None → legacy."""
    return lap.analyze_with_laptop_gate(
        frame, width, height, include_laptop_details=include_laptop_details
    )


def _legacy_detection_meta(dist: dict) -> tuple[str, bool]:
    """Distance path is not laptop detection; map screen heuristic vs distance TFLite for API contract."""
    src = dist.get("source", "none")
    if src == "edge_heuristic":
        return "heuristic", False
    if src == "tflite":
        conf = float(dist.get("confidence", 0.0))
        return "tflite", conf >= lap.LAPTOP_CONFIDENCE_MIN
    return "none", False


def _attach_detection(body: dict, source: str, reliable: bool) -> dict:
    out = dict(body)
    out["detection_source"] = source
    out["detection_reliable"] = bool(reliable)
    return out


@app.post("/api/distance")
def distance(req: FrameRequest):
    """Distance + brightness, or laptop-gated bbox when Roboflow workflow or objects_model.tflite is active."""
    frame = _decode_frame(req)
    gated = _gated_distance_body(frame, req.width, req.height, include_laptop_details=False)
    if gated is not None:
        if gated.get("laptop_detected") is False:
            return {
                k: gated[k]
                for k in ("laptop_detected", "detection_source", "detection_reliable")
                if k in gated
            }
        if not gated.get("analysis_ready", False):
            keys = (
                "laptop_detected",
                "analysis_ready",
                "reason",
                "laptop_class_id",
                "laptop_class",
                "laptop_confidence",
                "detection_source",
                "detection_reliable",
            )
            return {k: gated[k] for k in keys if k in gated}
        return gated
    dist = inf.predict_distance(frame, req.width, req.height)
    br = bright_ml.predict_brightness(frame, req.width, req.height)
    ds, dr = _legacy_detection_meta(dist)
    return _attach_detection({**dist, **br}, ds, dr)


@app.post("/api/analyze")
def analyze(
    req: FrameRequest,
    include_objects: bool = Query(
        False,
        description="If true, add laptop_top3 (and objects when legacy path) when applicable",
    ),
):
    """Distance + brightness; laptop-gated when Roboflow workflow or objects_model.tflite is active."""
    frame = _decode_frame(req)
    gated = _gated_distance_body(
        frame, req.width, req.height, include_laptop_details=include_objects
    )
    if gated is not None:
        if gated.get("laptop_detected") is False:
            keys = ("laptop_detected", "detection_source", "detection_reliable", "laptop_top3")
            return {k: gated[k] for k in keys if k in gated}
        if not gated.get("analysis_ready", False):
            keys = (
                "laptop_detected",
                "analysis_ready",
                "reason",
                "laptop_class_id",
                "laptop_class",
                "laptop_confidence",
                "laptop_top3",
                "detection_source",
                "detection_reliable",
            )
            return {k: gated[k] for k in keys if k in gated}
        return gated
    dist = inf.predict_distance(frame, req.width, req.height)
    br = bright_ml.predict_brightness(frame, req.width, req.height)
    ds, dr = _legacy_detection_meta(dist)
    out = _attach_detection({**dist, **br}, ds, dr)
    if include_objects:
        obj = obj_inf.predict_objects(frame)
        if obj is not None:
            out["objects"] = obj
    return out


@app.post("/api/objects")
def classify_objects(req: FrameRequest):
    """Run the laptop/object TFLite classifier (grayscale frame → RGB at model resolution)."""
    frame = _decode_frame(req)
    result = obj_inf.predict_objects(frame)
    if result is None:
        raise HTTPException(
            status_code=503,
            detail="objects model not loaded (missing models/objects_model.tflite)",
        )
    conf = float(result.get("confidence", 0.0))
    return _attach_detection(result, "tflite", conf >= lap.LAPTOP_CONFIDENCE_MIN)


@app.post("/api/collect")
def collect(req: CollectRequest):
    """Save a labeled frame to the training dataset."""
    frame = _decode_frame(req)
    result = col.save_frame(frame, req.distance_cm)
    return result


@app.get("/api/stats")
def stats():
    """Return dataset statistics (frame count per distance class)."""
    return col.get_stats()


@app.post("/api/reload")
def reload():
    """Hot-reload distance TFLite (models/model.tflite) and brightness TFLite (models/brightness_model.tflite)."""
    ok_d = inf.reload_model()
    ok_b = bright_ml.reload_brightness_model()
    return {
        "reloaded": ok_d,
        "model_loaded": inf._interp is not None,
        "brightness_reloaded": ok_b,
        "brightness_model_loaded": bright_ml.brightness_model_loaded(),
    }


@app.post("/api/reload-objects")
def reload_objects():
    """Hot-reload objects TFLite (models/objects_model.tflite) after train_objects.py."""
    ok = obj_inf.reload_objects_model()
    return {"reloaded": ok, "objects_model_loaded": obj_inf.objects_model_loaded()}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_frame(req: FrameRequest) -> np.ndarray:
    try:
        raw = base64.b64decode(req.frame)
    except Exception:
        raise HTTPException(status_code=400, detail="frame: invalid base64")

    expected = req.width * req.height
    if len(raw) != expected:
        raise HTTPException(
            status_code=400,
            detail=f"frame: expected {expected} bytes for {req.width}×{req.height}, got {len(raw)}",
        )

    return np.frombuffer(raw, dtype=np.uint8).reshape(req.height, req.width)


# ---------------------------------------------------------------------------
# Run with: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
