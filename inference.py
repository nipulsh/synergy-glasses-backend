"""
Distance inference from a camera frame captured by the OV2640 on the glasses.

The frame arrives as a raw 80×60 uint8 grayscale array (2× sub-sampled from
the native 160×120 QQVGA capture).

Inference priority
------------------
1. TFLite model at models/model.tflite  (trained via train.py)
2. Screen-size detection (bright screen vs dark background) — see screen_distance.py

Training
--------
Collect frames with the /api/collect endpoint, then run:
    cd backend && python train.py
"""

from __future__ import annotations

import numpy as np
import cv2
from pathlib import Path

from screen_distance import analyze_screen_distance

MODEL_PATH = Path(__file__).parent / "models" / "model.tflite"

# ---------------------------------------------------------------------------
# TFLite loader
# ---------------------------------------------------------------------------
_interp = None


def _load_tflite() -> bool:
    global _interp
    if not MODEL_PATH.exists():
        print("[inference] No TFLite model found — using screen-size detection fallback.")
        print(f"[inference] Train one with: cd backend && python train.py")
        return False
    try:
        try:
            import tflite_runtime.interpreter as tflite  # type: ignore
            _interp = tflite.Interpreter(model_path=str(MODEL_PATH))
        except ImportError:
            import tensorflow as tf  # type: ignore
            _interp = tf.lite.Interpreter(model_path=str(MODEL_PATH))
        _interp.allocate_tensors()
        print(f"[inference] TFLite model loaded: {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"[inference] Failed to load TFLite model: {e}")
        return False


_load_tflite()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_distance(frame: np.ndarray, width: int | None = None, height: int | None = None) -> dict:
    """
    Predict distance from an (H, W) uint8 grayscale frame captured by the
    OV2640 camera on the ESP32-S3 Sense glasses.

    Args:
        frame: (H, W) uint8 grayscale.
        width, height: Optional; default to frame.shape[1], frame.shape[0].

    Returns:
        {
            "distance_cm": float,
            "category":    "TOO_CLOSE" | "OK" | "FAR" | "UNKNOWN" (fallback only),
            "confidence":  float,   # 0–1
            "source":      "tflite" | "edge_heuristic",
            "screen_ratio": int,   # only when source is edge_heuristic
        }
    """
    if _interp is not None:
        return _predict_tflite(frame)
    h, w = int(frame.shape[0]), int(frame.shape[1])
    ww = int(width) if width is not None else w
    hh = int(height) if height is not None else h
    return analyze_screen_distance(frame, ww, hh)


def reload_model() -> bool:
    """Hot-reload the TFLite model after training. Returns True if successful."""
    return _load_tflite()


# ---------------------------------------------------------------------------
# TFLite inference
# ---------------------------------------------------------------------------

def _predict_tflite(frame: np.ndarray) -> dict:
    inp = _interp.get_input_details()
    out = _interp.get_output_details()

    in_shape = inp[0]["shape"]  # [1, H, W, C]
    h, w = int(in_shape[1]), int(in_shape[2])
    channels = int(in_shape[3]) if len(in_shape) == 4 else 1

    resized = cv2.resize(frame, (w, h))

    if channels == 1:
        tensor = resized.astype(np.float32)[np.newaxis, :, :, np.newaxis] / 255.0
    else:
        tensor = (
            cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB).astype(np.float32)[np.newaxis] / 255.0
        )

    _interp.set_tensor(inp[0]["index"], tensor)
    _interp.invoke()
    output = _interp.get_tensor(out[0]["index"])[0]

    # Regression: single float → distance_cm
    if output.size == 1:
        dist_cm = float(np.clip(output.flat[0], 10, 120))
        return {
            "distance_cm": round(dist_cm, 1),
            "category": _cm_to_category(dist_cm),
            "confidence": 0.87,
            "source": "tflite",
        }

    # Classification: [too_close, ok, far] logits
    probs = _softmax(output.astype(np.float32))
    classes = ["TOO_CLOSE", "OK", "FAR"]
    cm_map = {"TOO_CLOSE": 25.0, "OK": 50.0, "FAR": 80.0}
    idx = int(np.argmax(probs))
    cat = classes[idx]
    return {
        "distance_cm": cm_map[cat],
        "category": cat,
        "confidence": round(float(probs[idx]), 3),
        "source": "tflite",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cm_to_category(cm: float) -> str:
    if cm < 35:
        return "TOO_CLOSE"
    if cm > 65:
        return "FAR"
    return "OK"


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()
