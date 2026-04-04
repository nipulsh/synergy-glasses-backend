"""
OcuSmart Distance API

Endpoints
---------
GET  /api/health          → {"status": "ok", "model_loaded": bool}
POST /api/distance        → distance + brightness (same frame)
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

import brightness as bright
import collector as col
import inference as inf
import object_inference as obj_inf

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
    }


@app.post("/api/distance")
def distance(req: FrameRequest):
    """Distance (ML or heuristic) plus screen/ambient brightness from one frame."""
    frame = _decode_frame(req)
    dist = inf.predict_distance(frame, req.width, req.height)
    br = bright.analyze_brightness(frame, req.width, req.height)
    return {**dist, **br}


@app.post("/api/analyze")
def analyze(
    req: FrameRequest,
    include_objects: bool = Query(
        False,
        description="If true, add `objects` key when objects_model.tflite is loaded",
    ),
):
    """Distance (ML or heuristic) plus screen/ambient brightness from one frame."""
    frame = _decode_frame(req)
    dist = inf.predict_distance(frame, req.width, req.height)
    br = bright.analyze_brightness(frame, req.width, req.height)
    out = {**dist, **br}
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
    return result


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
    """Hot-reload distance TFLite (models/model.tflite)."""
    ok = inf.reload_model()
    return {"reloaded": ok, "model_loaded": inf._interp is not None}


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
