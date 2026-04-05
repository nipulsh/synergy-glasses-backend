"""EMA smoothing for distance and bbox fill ratio (single-device global state)."""

EMA_ALPHA = 0.4

_prev_distance_cm: float | None = None
_prev_ratio_pct: float | None = None


def apply_smoothing(distance_cm: float, bbox_area_ratio_pct: float) -> tuple[float, float]:
    global _prev_distance_cm, _prev_ratio_pct
    if _prev_distance_cm is None:
        _prev_distance_cm = float(distance_cm)
        _prev_ratio_pct = float(bbox_area_ratio_pct)
        return round(_prev_distance_cm, 2), round(_prev_ratio_pct, 2)
    _prev_distance_cm = EMA_ALPHA * float(distance_cm) + (1.0 - EMA_ALPHA) * _prev_distance_cm
    _prev_ratio_pct = EMA_ALPHA * float(bbox_area_ratio_pct) + (1.0 - EMA_ALPHA) * _prev_ratio_pct
    return round(_prev_distance_cm, 2), round(_prev_ratio_pct, 2)
