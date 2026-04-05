"""Save labeled grayscale frames for training datasets."""

import time
from pathlib import Path

import numpy as np

DATASET_ROOT = Path(__file__).resolve().parent / "dataset"


def _class_for_distance(distance_cm: float) -> str:
    if distance_cm < 35.0:
        return "too_close"
    if distance_cm > 65.0:
        return "far"
    return "ok"


def save_frame(frame: np.ndarray, distance_cm: float) -> dict:
    label = _class_for_distance(distance_cm)
    out_dir = DATASET_ROOT / label
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)
    path = out_dir / f"frame_{ts}.raw"
    path.write_bytes(frame.tobytes(order="C"))
    return {"status": "saved", "class": label, "path": str(path)}


def get_stats() -> dict:
    counts: dict[str, int] = {"too_close": 0, "ok": 0, "far": 0}
    if not DATASET_ROOT.is_dir():
        return counts
    for name in counts:
        d = DATASET_ROOT / name
        if d.is_dir():
            counts[name] = sum(1 for p in d.iterdir() if p.is_file())
    return counts
