"""
ML brightness: TFLite regressor at models/brightness_model.tflite (train with train_brightness.py).

Falls back to heuristic brightness.analyze_brightness when the file is missing.
Input matches distance inference: grayscale uint8 (H, W) → resize to model, gray→RGB if needed.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import brightness as bright

BRIGHTNESS_MODEL_PATH = Path(__file__).parent / "models" / "brightness_model.tflite"

_interp = None


def _load_tflite() -> bool:
    global _interp
    if not BRIGHTNESS_MODEL_PATH.exists():
        print("[brightness_ml] No brightness_model.tflite — using heuristic brightness.py")
        _interp = None
        return False
    try:
        try:
            import tflite_runtime.interpreter as tflite  # type: ignore
            _interp = tflite.Interpreter(model_path=str(BRIGHTNESS_MODEL_PATH))
        except ImportError:
            import tensorflow as tf  # type: ignore
            _interp = tf.lite.Interpreter(model_path=str(BRIGHTNESS_MODEL_PATH))
        _interp.allocate_tensors()
        print(f"[brightness_ml] TFLite loaded: {BRIGHTNESS_MODEL_PATH}")
        return True
    except Exception as e:
        print(f"[brightness_ml] Failed to load TFLite: {e}")
        _interp = None
        return False


_load_tflite()


def brightness_model_loaded() -> bool:
    return _interp is not None


def reload_brightness_model() -> bool:
    return _load_tflite()


def predict_brightness(frame: np.ndarray, width: int, height: int) -> dict:
    """
    Returns same keys as analyze_brightness plus brightness_source: tflite | heuristic.
    """
    if _interp is None:
        out = bright.analyze_brightness(frame, width, height)
        out["brightness_source"] = "heuristic"
        return out

    inp = _interp.get_input_details()
    out_d = _interp.get_output_details()
    in_shape = inp[0]["shape"]
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
    raw = _interp.get_tensor(out_d[0]["index"])[0].astype(np.float32)

    # Model outputs sigmoid → [0,1]; scale to percent
    screen_01 = float(np.clip(raw[0], 0.0, 1.0))
    ambient_01 = float(np.clip(raw[1], 0.0, 1.0))
    screen_br = screen_01 * 100.0
    ambient_br = ambient_01 * 100.0

    out = bright.pack_brightness(screen_br, ambient_br)
    out["brightness_source"] = "tflite"
    return out
