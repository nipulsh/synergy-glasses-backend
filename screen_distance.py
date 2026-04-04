"""
Screen-size distance from an 80×60 grayscale frame: bright laptop screen vs dark surround.

Exposes find_largest_bright_bbox for laptop-gated bbox distance + brightness split.

Calibration (env): DISTANCE_D_REF, DISTANCE_REF_RATIO_PCT — inverse square-root model:
  distance_cm = D_ref * sqrt(ref_ratio / current_ratio)
with current_ratio the same units as ref (bbox / contour fill as percentage of frame area).
"""

from __future__ import annotations

import os
from typing import Any, Optional, Tuple

import cv2
import numpy as np

# Reference distance (cm) at reference fill ratio; tune with captured calibration frames.
DISTANCE_D_REF = float(os.environ.get("DISTANCE_D_REF", "60.0"))
# Fill ratio (% of frame area) at D_ref; must match how current_ratio is computed (0–100 scale).
DISTANCE_REF_RATIO_PCT = float(os.environ.get("DISTANCE_REF_RATIO_PCT", "25.0"))
# Minimum divisor to avoid blow-ups when the blob is tiny or zero.
DISTANCE_MIN_RATIO_PCT = float(os.environ.get("DISTANCE_MIN_RATIO_PCT", "0.5"))
DISTANCE_CM_MIN = float(os.environ.get("DISTANCE_CM_MIN", "20.0"))
DISTANCE_CM_MAX = float(os.environ.get("DISTANCE_CM_MAX", "150.0"))


def _otsu_largest_contour(
    frame: np.ndarray, width: int, height: int
) -> Optional[Tuple[Any, Tuple[int, int, int, int], float]]:
    """
    Returns (contour, (x,y,bw,bh), rect_confidence) or None.
    """
    h, w = int(frame.shape[0]), int(frame.shape[1])
    total_area = float(width * height)
    min_contour_area = max(50.0, total_area * 0.01)

    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    fg = blurred[binary == 255]
    bg = blurred[binary == 0]
    if fg.size and bg.size and float(np.mean(fg)) < float(np.mean(bg)):
        binary = 255 - binary

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest))
    if area < min_contour_area:
        return None

    x, y, bw, bh = cv2.boundingRect(largest)
    short_side = min(bw, bh)
    long_side = max(bw, bh)
    aspect = (long_side / short_side) if short_side > 0 else 0.0
    rectangular = 1.2 <= aspect <= 2.5
    cref = 0.80 if rectangular else 0.50
    return (largest, (x, y, bw, bh), cref)


def find_largest_bright_bbox(
    frame: np.ndarray, width: int, height: int
) -> Optional[Tuple[int, int, int, int, float, float]]:
    """
    Returns (x, y, bw, bh, bbox_area_ratio_pct, aspect_confidence).
    bbox_area_ratio_pct = 100 * (bw*bh) / (W*H).
    """
    if frame.ndim != 2:
        raise ValueError("frame must be 2D (H, W)")
    h, w = int(frame.shape[0]), int(frame.shape[1])
    if h != height or w != width:
        raise ValueError(f"frame shape ({h}, {w}) does not match height={height}, width={width}")

    got = _otsu_largest_contour(frame, width, height)
    if got is None:
        return None
    _, (x, y, bw, bh), cref = got
    if bw <= 0 or bh <= 0:
        return None
    total_area = float(width * height)
    bbox_area_ratio_pct = (float(bw * bh) / total_area) * 100.0
    return (x, y, bw, bh, bbox_area_ratio_pct, cref)


def category_from_distance_cm(distance_cm: float) -> str:
    if distance_cm < 35.0:
        return "TOO_CLOSE"
    if distance_cm > 65.0:
        return "FAR"
    return "OK"


def compute_distance_from_fill_ratio_pct(
    area_ratio_pct: float,
    rect_confidence: float,
    *,
    source: str = "laptop_bbox",
) -> dict[str, Any]:
    """
    Calibrated model: distance_cm = D_ref * sqrt(ref_ratio / current_ratio).
    Safe denominator, clamped to [DISTANCE_CM_MIN, DISTANCE_CM_MAX].
    """
    ratio_pct = float(area_ratio_pct)
    denom = max(ratio_pct, DISTANCE_MIN_RATIO_PCT)
    ref = max(float(DISTANCE_REF_RATIO_PCT), DISTANCE_MIN_RATIO_PCT)
    distance_cm = float(DISTANCE_D_REF) * float(np.sqrt(ref / denom))
    distance_cm = float(np.clip(distance_cm, DISTANCE_CM_MIN, DISTANCE_CM_MAX))
    distance_cm = round(distance_cm, 1)
    screen_ratio = int(round(np.clip(ratio_pct, 0.0, 100.0)))
    return {
        "distance_cm": distance_cm,
        "category": category_from_distance_cm(distance_cm),
        "confidence": rect_confidence,
        "source": source,
        "screen_ratio": screen_ratio,
        "bbox_area_ratio_pct": round(ratio_pct, 2),
    }


def distance_from_bbox_ratio(bbox_area_ratio_pct: float, rect_confidence: float) -> dict:
    """Distance from laptop/screen bounding-box fill (% of frame area)."""
    return compute_distance_from_fill_ratio_pct(
        bbox_area_ratio_pct, rect_confidence, source="laptop_bbox"
    )


# Readable alias for laptop pipeline modules
compute_distance = distance_from_bbox_ratio


def analyze_screen_distance(frame: np.ndarray, width: int, height: int) -> dict:
    """Edge heuristic using contour-area ratio (legacy path when no laptop gate)."""
    h, w = int(frame.shape[0]), int(frame.shape[1])
    if frame.ndim != 2:
        raise ValueError("frame must be 2D (H, W)")
    if h != height or w != width:
        raise ValueError(f"frame shape ({h}, {w}) does not match height={height}, width={width}")

    got = _otsu_largest_contour(frame, width, height)
    if got is None:
        return _unknown_result()

    largest, (x, y, bw, bh), confidence = got
    total_area = float(width * height)
    ratio_pct = (float(cv2.contourArea(largest)) / total_area) * 100.0
    out = compute_distance_from_fill_ratio_pct(ratio_pct, confidence, source="edge_heuristic")
    # Legacy key: no bbox_area_ratio_pct clutter name for contour path — keep screen_ratio only
    out.pop("bbox_area_ratio_pct", None)
    return out


def _unknown_result() -> dict:
    return {
        "distance_cm": round(float(np.clip(50.0, DISTANCE_CM_MIN, DISTANCE_CM_MAX)), 1),
        "category": "UNKNOWN",
        "confidence": 0.20,
        "source": "edge_heuristic",
        "screen_ratio": 0,
    }
