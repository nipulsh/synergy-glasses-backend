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
  "height":   60,
  "roboflow_detection_frame": null,  // optional: Roboflow-only image; `frame` = latest for metrics
  "laptop_step_token": null        // optional: from POST /api/laptop-gate/roboflow
}
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

from dotenv import load_dotenv

# Load repo-root `.env` before imports that read os.environ at module load (e.g. laptop_scene).
_APP_DIR = Path(__file__).resolve().parent
load_dotenv(_APP_DIR / ".env")

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
import screen_distance as sd

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
    roboflow_detection_frame: str | None = Field(
        None,
        description="Optional: image sent to Roboflow only; `frame` is used for distance/brightness (same dimensions).",
    )
    laptop_step_token: str | None = Field(
        None,
        description="From POST /api/laptop-gate/roboflow; use with latest `frame` to finish analysis without re-calling Roboflow.",
    )


class CollectRequest(FrameRequest):
    distance_cm: float = Field(..., ge=5, le=200, description="True distance in centimetres")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def _positive_int_env(key: str, default: int) -> int:
    raw = os.environ.get(key, str(default))
    try:
        v = int(str(raw).strip())
    except ValueError:
        return default
    return max(1, v)


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
        # Hints for mobile/UI: server does not enforce timing; client polls /api/* and applies alerts.
        "client_data_poll_interval_sec": _positive_int_env("CLIENT_DATA_POLL_INTERVAL_SEC", 2),
        "client_alert_cooldown_sec": _positive_int_env("CLIENT_ALERT_COOLDOWN_SEC", 20),
    }


def _gated_distance_body(
    frame: np.ndarray,
    width: int,
    height: int,
    include_laptop_details: bool,
    *,
    include_workflow_annotated_image: bool = False,
    roboflow_detection_frame: np.ndarray | None = None,
    cached_laptop_detection: tuple[dict, str, bool] | None = None,
):
    """If Roboflow workflow gate is active: laptop-only pipeline (TFLite laptop gate off in laptop_scene); else None → legacy."""
    return lap.analyze_with_laptop_gate(
        frame,
        width,
        height,
        include_laptop_details=include_laptop_details,
        include_workflow_annotated_image=include_workflow_annotated_image,
        roboflow_detection_frame=roboflow_detection_frame,
        cached_laptop_detection=cached_laptop_detection,
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


def _resolve_laptop_gate_options(req: FrameRequest) -> tuple[np.ndarray | None, tuple[dict, str, bool] | None]:
    """Optional Roboflow-only detection frame; or consume step token (skips Roboflow)."""
    tok = req.laptop_step_token
    if tok is not None and str(tok).strip():
        det = rwf.consume_laptop_step_token(str(tok).strip(), req.width, req.height)
        if det is None:
            raise HTTPException(
                status_code=400,
                detail="laptop_step_token invalid, expired, or width/height mismatch",
            )
        cached: tuple[dict, str, bool] = (
            det,
            "roboflow",
            lap._detection_reliable_for_classifier(det, "roboflow"),
        )
        return None, cached
    rf: str | None = req.roboflow_detection_frame
    if rf is not None and str(rf).strip():
        return _decode_frame_b64(str(rf).strip(), req.width, req.height), None
    return None, None


@app.post("/api/laptop-gate/roboflow")
def laptop_gate_roboflow(req: FrameRequest):
    """
    Run only the Roboflow workflow (blocking until the API responds). Returns a short-lived
    `step_token` when a laptop is detected; send the latest glasses frame to POST /api/distance
    (or /api/analyze) with that token to compute distance/brightness without re-calling Roboflow.
    """
    if not (rwf.gate_enabled() and rwf.configured()):
        raise HTTPException(
            status_code=400,
            detail="Roboflow laptop gate is not enabled or not configured",
        )
    frame = _decode_frame(req)
    det = rwf.try_predict_for_gate(frame, req.width, req.height)
    reliable = lap._detection_reliable_for_classifier(det, "roboflow") if det else False
    if det is None:
        return {
            "laptop_detected": False,
            "reason": "roboflow_unavailable",
            "detection_source": "roboflow",
            "detection_reliable": False,
        }
    if not lap._is_laptop(det):
        return {
            "laptop_detected": False,
            "detection_source": "roboflow",
            "detection_reliable": reliable,
            "laptop_top3": det.get("top3"),
        }
    # Omit large `annotated_image` from the step token payload (kept in this HTTP response only).
    det_for_token = {k: v for k, v in det.items() if k != "annotated_image"}
    step_token = rwf.issue_laptop_step_token(det_for_token, req.width, req.height)
    out = {
        "laptop_detected": True,
        "step_token": step_token,
        "laptop_class_id": det["class_id"],
        "laptop_class": det["class_name"],
        "laptop_confidence": det["confidence"],
        "detection_source": "roboflow",
        "detection_reliable": reliable,
        "laptop_top3": det.get("top3"),
    }
    if det.get("source") == "roboflow_workflow":
        out["laptop_source"] = "roboflow_workflow"
    wf_dist = det.get("workflow_distance_cm")
    if wf_dist is not None:
        try:
            out["distance_cm"] = round(
                float(
                    np.clip(
                        float(wf_dist),
                        sd.DISTANCE_CM_MIN,
                        sd.DISTANCE_CM_MAX,
                    )
                ),
                1,
            )
        except (TypeError, ValueError):
            pass
    ann = det.get("annotated_image")
    if isinstance(ann, str) and ann.strip():
        out["annotated_image"] = ann.strip()
    return out


@app.post("/api/distance")
def distance(req: FrameRequest):
    """Distance + brightness, or laptop-gated bbox when Roboflow workflow is active (TFLite laptop gate commented out)."""
    frame = _decode_frame(req)
    rf, cached = _resolve_laptop_gate_options(req)
    gated = _gated_distance_body(
        frame,
        req.width,
        req.height,
        include_laptop_details=False,
        roboflow_detection_frame=rf,
        cached_laptop_detection=cached,
    )
    if gated is None:
        return {"laptop_detected": False, "reason": "no_gate_configured", "detection_source": "none", "detection_reliable": False}
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


@app.post("/api/analyze")
def analyze(
    req: FrameRequest,
    include_objects: bool = Query(
        False,
        description="If true, add laptop_top3 (and objects when legacy path) when applicable",
    ),
):
    """Distance + brightness; laptop-gated when Roboflow workflow is active (TFLite laptop gate commented out)."""
    frame = _decode_frame(req)
    rf, cached = _resolve_laptop_gate_options(req)
    gated = _gated_distance_body(
        frame,
        req.width,
        req.height,
        include_laptop_details=include_objects,
        include_workflow_annotated_image=True,
        roboflow_detection_frame=rf,
        cached_laptop_detection=cached,
    )
    if gated is None:
        return {"laptop_detected": False, "reason": "no_gate_configured", "detection_source": "none", "detection_reliable": False}
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

def _decode_frame_b64(frame_b64: str, width: int, height: int) -> np.ndarray:
    try:
        raw = base64.b64decode(frame_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="frame: invalid base64")

    expected = width * height
    if len(raw) != expected:
        raise HTTPException(
            status_code=400,
            detail=f"frame: expected {expected} bytes for {width}×{height}, got {len(raw)}",
        )

    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width)


def _decode_frame(req: FrameRequest) -> np.ndarray:
    return _decode_frame_b64(req.frame, req.width, req.height)


# ---------------------------------------------------------------------------
# Run with: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
