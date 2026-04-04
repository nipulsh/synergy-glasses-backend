"""
Laptop / object classifier: TFLite from train_objects.py (Kaggle laptop dataset).

Input: same grayscale uint8 (H, W) as the glasses API — resized to model input,
replicated to RGB, normalized [0, 1]. Training used 160×160; the interpreter
reads actual H×W from the flatbuffer.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

OBJECTS_MODEL_PATH = Path(__file__).parent / "models" / "objects_model.tflite"
CLASS_NAMES_PATH = Path(__file__).parent / "models" / "objects_class_names.json"

_interp = None
_class_names: list[str] = []


def _load_class_names() -> list[str]:
    if not CLASS_NAMES_PATH.exists():
        return []
    try:
        data = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
        return list(data.get("class_names", []))
    except Exception:
        return []


def _load_tflite() -> bool:
    global _interp, _class_names
    _class_names = _load_class_names()
    if not OBJECTS_MODEL_PATH.exists():
        print("[objects] No objects_model.tflite — POST /api/objects will return 503.")
        _interp = None
        return False
    try:
        try:
            import tflite_runtime.interpreter as tflite  # type: ignore
            _interp = tflite.Interpreter(model_path=str(OBJECTS_MODEL_PATH))
        except ImportError:
            import tensorflow as tf  # type: ignore
            _interp = tf.lite.Interpreter(model_path=str(OBJECTS_MODEL_PATH))
        _interp.allocate_tensors()
        print(f"[objects] TFLite loaded: {OBJECTS_MODEL_PATH}")
        return True
    except Exception as e:
        print(f"[objects] Failed to load TFLite: {e}")
        _interp = None
        return False


_load_tflite()


def objects_model_loaded() -> bool:
    return _interp is not None


def reload_objects_model() -> bool:
    """Hot-reload after retraining. Returns True if a model is loaded."""
    return _load_tflite()


def predict_objects(frame: np.ndarray) -> dict | None:
    """
    Classify from (H, W) uint8 grayscale. Returns None if no model loaded.

    Response keys are stable for clients: class_id, class_name, confidence, top3.
    """
    if _interp is None:
        return None

    inp = _interp.get_input_details()
    out = _interp.get_output_details()
    in_shape = inp[0]["shape"]
    h, w = int(in_shape[1]), int(in_shape[2])
    channels = int(in_shape[3]) if len(in_shape) == 4 else 1

    resized = cv2.resize(frame, (w, h))
    if channels == 1:
        tensor = resized.astype(np.float32)[np.newaxis, :, :, np.newaxis] / 255.0
    else:
        rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB).astype(np.float32)[np.newaxis] / 255.0
        tensor = rgb

    _interp.set_tensor(inp[0]["index"], tensor)
    _interp.invoke()
    raw = _interp.get_tensor(out[0]["index"])[0].astype(np.float32)

    # Softmax if logits; otherwise treat as probabilities
    s = float(raw.sum())
    if s <= 0 or raw.max() > 1.01 or s < 0.5:
        probs = _softmax(raw)
    else:
        probs = raw / s

    idx = int(np.argmax(probs))
    n = len(probs)

    def _name(i: int) -> str:
        if 0 <= i < len(_class_names):
            return _class_names[i]
        return str(i)

    top_idx = np.argsort(-probs)[: min(3, n)]
    top3 = [
        {
            "class_id": int(i),
            "class_name": _name(int(i)),
            "confidence": round(float(probs[int(i)]), 4),
        }
        for i in top_idx
    ]

    return {
        "class_id": idx,
        "class_name": _name(idx),
        "confidence": round(float(probs[idx]), 4),
        "top3": top3,
        "source": "objects_tflite",
    }


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()
