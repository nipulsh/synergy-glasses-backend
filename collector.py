"""
Dataset management for training data collection.

Saves labeled frames to:  backend/dataset/{label_cm}cm/{timestamp}.jpg
"""

from __future__ import annotations

import cv2
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

DATASET_DIR = Path(__file__).parent / "dataset"


def save_frame(frame: np.ndarray, distance_cm: float) -> dict:
    """
    Save a grayscale frame with its distance label.

    Args:
        frame:       H×W uint8 grayscale numpy array (typically 60×80).
        distance_cm: True distance from user's eyes to screen in centimetres.

    Returns:
        {"path": str, "label_cm": int, "total_in_class": int}
    """
    # Round to nearest 5 cm bin so labels stay clean
    label = int(round(distance_cm / 5.0) * 5)
    out_dir = DATASET_DIR / f"{label}cm"
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    out_path = out_dir / f"{ts}.jpg"

    # Upscale to 160×120 before saving (better quality for training)
    big = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(str(out_path), big, [cv2.IMWRITE_JPEG_QUALITY, 90])

    total = len(list(out_dir.glob("*.jpg")))
    return {
        "path": str(out_path.relative_to(Path(__file__).parent)),
        "label_cm": label,
        "total_in_class": total,
    }


def get_stats() -> dict:
    """Return dataset statistics."""
    if not DATASET_DIR.exists():
        return {"total_frames": 0, "classes": {}}

    classes: dict[str, int] = {}
    total = 0
    for label_dir in sorted(DATASET_DIR.iterdir()):
        if not label_dir.is_dir():
            continue
        count = len(list(label_dir.glob("*.jpg")))
        classes[label_dir.name] = count
        total += count

    return {"total_frames": total, "classes": classes}
