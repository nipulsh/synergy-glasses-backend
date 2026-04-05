"""Distance from laptop/screen bbox fill ratio and bright-blob fallback."""

import math
import os

import cv2
import numpy as np

DISTANCE_D_REF = float(os.environ.get("DISTANCE_D_REF", "60.0"))
DISTANCE_REF_RATIO_PCT = float(os.environ.get("DISTANCE_REF_RATIO_PCT", "25.0"))
DISTANCE_MIN_RATIO_PCT = float(os.environ.get("DISTANCE_MIN_RATIO_PCT", "0.5"))
DISTANCE_CM_MIN = float(os.environ.get("DISTANCE_CM_MIN", "20.0"))
DISTANCE_CM_MAX = float(os.environ.get("DISTANCE_CM_MAX", "150.0"))


def category_from_distance_cm(cm: float) -> str:
    if cm < 35.0:
        return "TOO_CLOSE"
    if cm > 65.0:
        return "FAR"
    return "OK"


def _cref_from_aspect(bw: int, bh: int) -> float:
    if bh <= 0:
        return 0.50
    ar = bw / float(bh)
    if 1.2 <= ar <= 2.5:
        return 0.80
    return 0.50


def find_largest_bright_bbox(
    frame: np.ndarray, width: int, height: int
) -> tuple[int, int, int, int, float, float] | None:
    if frame.size == 0 or width <= 0 or height <= 0:
        return None
    gray = frame.reshape((height, width)) if frame.ndim == 1 else frame
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg_m = binary > 0
    bg_m = binary == 0
    if np.any(fg_m) and np.any(bg_m):
        if float(np.mean(blurred[fg_m])) < float(np.mean(blurred[bg_m])):
            binary = cv2.bitwise_not(binary)
    min_area = max(50, int(0.01 * width * height))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area <= best_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        best = (x, y, bw, bh)
        best_area = area
    if best is None:
        return None
    x, y, bw, bh = best
    bbox_area_ratio_pct = 100.0 * (bw * bh) / float(width * height)
    cref = _cref_from_aspect(bw, bh)
    return x, y, bw, bh, bbox_area_ratio_pct, cref


def compute_distance(bbox_area_ratio_pct: float, cref: float) -> dict:
    ratio = max(float(bbox_area_ratio_pct), DISTANCE_MIN_RATIO_PCT)
    distance_cm = DISTANCE_D_REF * math.sqrt(DISTANCE_REF_RATIO_PCT / ratio)
    distance_cm = max(DISTANCE_CM_MIN, min(DISTANCE_CM_MAX, distance_cm))
    category = category_from_distance_cm(distance_cm)
    screen_ratio = int(round(bbox_area_ratio_pct))
    return {
        "distance_cm": float(distance_cm),
        "category": category,
        "confidence": cref,
        "source": "laptop_bbox",
        "screen_ratio": screen_ratio,
        "bbox_area_ratio_pct": float(bbox_area_ratio_pct),
    }
