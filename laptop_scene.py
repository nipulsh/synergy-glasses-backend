"""
When objects_model.tflite is loaded OR Roboflow is configured (fallback): gate on laptop + screen blob.

Pipeline (strict):
1. Decode frame (caller).
2. TFLite object classifier if models/objects_model.tflite is loaded — primary; Roboflow is not called per request in that case.
3. If no laptop → { laptop_detected: false }.
4. If laptop → Otsu bbox, inverse-sqrt distance, bbox-split brightness, EMA smoothing.

Roboflow runs only when the TFLite objects model is absent but LAPTOP_USE_ROBOFLOW_WORKFLOW is on.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

import brightness as bright
import object_inference as obj_inf
import roboflow_workflow as rwf
import screen_distance as sd
import temporal_smoothing as tsmooth

_LAPTOP_NAME_HINTS = frozenset(
    ("laptop", "notebook", "monitor", "screen", "lcd", "computer", "display")
)


def _laptop_class_ids() -> frozenset[int] | None:
    raw = os.environ.get("LAPTOP_CLASS_IDS")
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s == "*":
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    ids: list[int] = []
    for p in parts:
        if p.lstrip("-").isdigit():
            ids.append(int(p))
    return frozenset(ids)


# Default 0.4 — env may set 0.35–0.45; below threshold counts as no laptop.
LAPTOP_CONFIDENCE_MIN = float(os.environ.get("LAPTOP_CONFIDENCE_MIN", "0.4"))


def _name_suggests_laptop(class_name: str) -> bool:
    n = class_name.lower().replace("_", " ")
    return any(h in n for h in _LAPTOP_NAME_HINTS)


def _is_laptop(det: dict) -> bool:
    conf = float(det.get("confidence", 0.0))
    if conf < LAPTOP_CONFIDENCE_MIN:
        return False
    lids = _laptop_class_ids()
    if lids is None:
        return True
    cid = int(det.get("class_id", -1))
    if cid in lids:
        return True
    return _name_suggests_laptop(str(det.get("class_name", "")))


def _detection_reliable_for_classifier(det: dict, source: str) -> bool:
    conf = float(det.get("confidence", 0.0))
    if source == "roboflow":
        return True
    if source == "tflite":
        return conf >= LAPTOP_CONFIDENCE_MIN
    return False


def detect_laptop(
    frame: np.ndarray,
    width: int,
    height: int,
) -> tuple[dict[str, Any], str, bool] | None:
    """
    Returns (det_dict, detection_source, detection_reliable), or None → use legacy distance path.

    TFLite is primary when loaded; Roboflow only if objects model is missing.
    """
    if obj_inf.objects_model_loaded():
        det = obj_inf.predict_objects(frame)
        if det is None:
            return None
        src = "tflite"
        return det, src, _detection_reliable_for_classifier(det, src)

    if rwf.gate_enabled() and rwf.configured():
        det = rwf.try_predict_for_gate(frame, width, height)
        if det is None:
            return None
        src = "roboflow"
        return det, src, _detection_reliable_for_classifier(det, src)

    return None


def compute_brightness(
    frame: np.ndarray,
    x: int,
    y: int,
    bw: int,
    bh: int,
    width: int,
    height: int,
) -> dict[str, Any]:
    """Screen = mean inside bbox; ambient = mean outside (corner fallback if bbox fills frame)."""
    return bright.analyze_brightness_bbox(frame, x, y, bw, bh, width, height)


def analyze_with_laptop_gate(
    frame: np.ndarray,
    width: int,
    height: int,
    *,
    include_laptop_details: bool = False,
) -> dict[str, Any] | None:
    """
    Returns None if neither objects TFLite nor Roboflow gate is available.

    Otherwise returns a dict with detection_source / detection_reliable on every branch.
    """
    got = detect_laptop(frame, width, height)
    if got is None:
        return None

    det, detection_source, detection_reliable = got

    if not _is_laptop(det):
        out: dict[str, Any] = {
            "laptop_detected": False,
            "detection_source": detection_source,
            "detection_reliable": detection_reliable,
        }
        if include_laptop_details:
            out["laptop_top3"] = det.get("top3")
        return out

    bbox = sd.find_largest_bright_bbox(frame, width, height)
    if bbox is None:
        out = {
            "laptop_detected": True,
            "analysis_ready": False,
            "reason": "no_screen_blob",
            "laptop_class_id": det["class_id"],
            "laptop_class": det["class_name"],
            "laptop_confidence": det["confidence"],
            "detection_source": detection_source,
            "detection_reliable": detection_reliable,
        }
        if include_laptop_details:
            out["laptop_top3"] = det.get("top3")
        return out

    x, y, bw, bh, bbox_ratio, cref = bbox
    dist_raw = sd.compute_distance(bbox_ratio, cref)
    raw_d = float(dist_raw["distance_cm"])
    raw_r = float(dist_raw.get("bbox_area_ratio_pct", bbox_ratio))
    sm_d, sm_r = tsmooth.apply_smoothing(raw_d, raw_r)

    dist_raw["distance_cm"] = sm_d
    dist_raw["category"] = sd.category_from_distance_cm(sm_d)
    dist_raw["screen_ratio"] = int(round(float(np.clip(sm_r, 0.0, 100.0))))
    dist_raw["bbox_area_ratio_pct"] = sm_r

    br = compute_brightness(frame, x, y, bw, bh, width, height)

    merged: dict[str, Any] = {
        "laptop_detected": True,
        "analysis_ready": True,
        "laptop_class_id": det["class_id"],
        "laptop_class": det["class_name"],
        "laptop_confidence": det["confidence"],
        "bbox": {"x": int(x), "y": int(y), "w": int(bw), "h": int(bh)},
        "brightness_source": "laptop_bbox",
        "detection_source": detection_source,
        "detection_reliable": detection_reliable,
        **dist_raw,
        **br,
    }
    if include_laptop_details:
        merged["laptop_top3"] = det.get("top3")
        if det.get("source") == "roboflow_workflow":
            merged["laptop_source"] = "roboflow_workflow"
    return merged
