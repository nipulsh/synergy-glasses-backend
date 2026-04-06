"""Save labeled grayscale frames and glasses camera captures under images/."""

import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np

IMAGES_ROOT = Path(__file__).resolve().parent / "images"
GLASSES_DIR = IMAGES_ROOT / "glasses"


def resolved_save_paths() -> dict[str, str]:
    """Absolute paths where glasses PNGs and labeled raw frames are written."""
    return {
        "images_root": str(IMAGES_ROOT.resolve()),
        "glasses_dir": str(GLASSES_DIR.resolve()),
    }

_seq_lock = threading.Lock()
_glasses_seq = 0
_labeled_seq = 0


def _unique_glasses_name(w: int, h: int) -> str:
    global _glasses_seq
    ns = time.time_ns()
    with _seq_lock:
        _glasses_seq += 1
        n = _glasses_seq
    return f"frame_{ns}_{n}_{w}x{h}.png"


def _unique_labeled_name() -> str:
    global _labeled_seq
    ns = time.time_ns()
    with _seq_lock:
        _labeled_seq += 1
        n = _labeled_seq
    return f"frame_{ns}_{n}.raw"


def _glasses_png_pixels_u8(gray: np.ndarray) -> np.ndarray:
    """Simple min/max linear stretch to 0–255 for PNG preview."""
    g = np.ascontiguousarray(gray, dtype=np.uint8)
    raw = (os.environ.get("GLASSES_PNG_RAW") or "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if raw:
        return g
    f = g.astype(np.float32)
    lo, hi = float(f.min()), float(f.max())
    if hi <= lo:
        return g
    out = ((f - lo) * (255.0 / (hi - lo))).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(out)


def _class_for_distance(distance_cm: float) -> str:
    if distance_cm < 35.0:
        return "too_close"
    if distance_cm > 65.0:
        return "far"
    return "ok"


def save_glasses_frame(frame: np.ndarray, width: int, height: int) -> dict:
    """Persist glasses frame as PNG under images/glasses/ (color BGR or stretched gray)."""
    GLASSES_DIR.mkdir(parents=True, exist_ok=True)
    w, h = int(width), int(height)
    path = GLASSES_DIR / _unique_glasses_name(w, h)
    if frame.ndim == 3 and frame.shape == (h, w, 3):
        ok = cv2.imwrite(str(path), frame)
    else:
        gray = frame.reshape((h, w)) if frame.ndim == 1 else frame
        if gray.shape != (h, w):
            gray = gray.reshape((h, w))
        png_gray = _glasses_png_pixels_u8(gray)
        ok = cv2.imwrite(str(path), png_gray)
    if not ok:
        raise OSError(f"cv2.imwrite failed for {path}")
    return {"status": "saved", "path": str(path.resolve())}


def save_frame(frame: np.ndarray, distance_cm: float) -> dict:
    label = _class_for_distance(distance_cm)
    out_dir = IMAGES_ROOT / label
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / _unique_labeled_name()
    path.write_bytes(frame.tobytes(order="C"))
    return {"status": "saved", "class": label, "path": str(path)}


def get_stats() -> dict:
    counts: dict[str, int] = {"glasses": 0, "too_close": 0, "ok": 0, "far": 0}
    if GLASSES_DIR.is_dir():
        counts["glasses"] = sum(
            1 for p in GLASSES_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".png"
        )
    if not IMAGES_ROOT.is_dir():
        return counts
    for name in ("too_close", "ok", "far"):
        d = IMAGES_ROOT / name
        if d.is_dir():
            counts[name] = sum(1 for p in d.iterdir() if p.is_file())
    return counts
