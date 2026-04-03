"""
Screen-size distance from an 80×60 grayscale frame: bright laptop screen vs dark surround.

The camera on the glasses points at the screen; a larger bright region implies the user
is closer. Used as the ML fallback when no TFLite model is loaded.
"""

from __future__ import annotations

import cv2
import numpy as np


def analyze_screen_distance(frame: np.ndarray, width: int, height: int) -> dict:
    """
    Args:
        frame: Raw grayscale uint8, shape (height, width).
        width, height: Expected dimensions (validated against frame.shape).

    Returns:
        distance_cm, category, confidence, source ("edge_heuristic"), screen_ratio (0–100 int).
    """
    if frame.ndim != 2:
        raise ValueError("frame must be 2D (H, W)")
    h, w = int(frame.shape[0]), int(frame.shape[1])
    if h != height or w != width:
        raise ValueError(f"frame shape ({h}, {w}) does not match height={height}, width={width}")

    total_area = float(width * height)
    min_contour_area = max(50.0, total_area * 0.01)

    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _unknown_result()

    largest = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest))
    if area < min_contour_area:
        return _unknown_result()

    ratio_pct = (area / total_area) * 100.0
    screen_ratio = int(round(np.clip(ratio_pct, 0.0, 100.0)))

    x, y, bw, bh = cv2.boundingRect(largest)
    short_side = min(bw, bh)
    long_side = max(bw, bh)
    aspect = (long_side / short_side) if short_side > 0 else 0.0
    rectangular = 1.2 <= aspect <= 2.5
    confidence = 0.80 if rectangular else 0.50

    distance_cm = float(np.clip(120.0 - ratio_pct * 1.2, 15.0, 120.0))
    distance_cm = round(distance_cm, 1)

    if ratio_pct >= 70.0:
        category = "TOO_CLOSE"
    elif ratio_pct >= 25.0:
        category = "OK"
    else:
        category = "FAR"

    return {
        "distance_cm": distance_cm,
        "category": category,
        "confidence": confidence,
        "source": "edge_heuristic",
        "screen_ratio": screen_ratio,
    }


def _unknown_result() -> dict:
    return {
        "distance_cm": 50,
        "category": "UNKNOWN",
        "confidence": 0.20,
        "source": "edge_heuristic",
        "screen_ratio": 0,
    }
