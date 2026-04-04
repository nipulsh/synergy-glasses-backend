"""
Exponential moving average for laptop-gated distance and bbox fill ratio.

Global state suits a single glasses device; alpha fixed per product spec.
"""

from __future__ import annotations

EMA_ALPHA = 0.4

_prev_distance_cm: float | None = None
_prev_bbox_ratio_pct: float | None = None


def apply_smoothing(distance_cm: float, bbox_area_ratio_pct: float) -> tuple[float, float]:
    """EMA (alpha=0.4) on raw distance and bbox fill %; first frame seeds the filter."""
    global _prev_distance_cm, _prev_bbox_ratio_pct
    if _prev_distance_cm is None:
        d = float(distance_cm)
    else:
        d = EMA_ALPHA * float(distance_cm) + (1.0 - EMA_ALPHA) * _prev_distance_cm
    if _prev_bbox_ratio_pct is None:
        r = float(bbox_area_ratio_pct)
    else:
        r = EMA_ALPHA * float(bbox_area_ratio_pct) + (1.0 - EMA_ALPHA) * _prev_bbox_ratio_pct
    _prev_distance_cm = d
    _prev_bbox_ratio_pct = r
    return round(d, 1), round(r, 2)


def reset_smoothing_state() -> None:
    """Optional: call on session change if multi-device support is added later."""
    global _prev_distance_cm, _prev_bbox_ratio_pct
    _prev_distance_cm = None
    _prev_bbox_ratio_pct = None
