"""
Gate on laptop + ROI when Roboflow workflow is enabled and configured.

Pipeline (strict):
1. Decode frame (caller).
2. Roboflow workflow (LAPTOP_USE_ROBOFLOW_WORKFLOW + ROBOFLOW_*).
3. If no laptop → { laptop_detected: false }.
4. If laptop → ROI for distance/brightness:
   - Roboflow workflow path: use workflow detection bbox when present (label matches ROBOFLOW_LAPTOP_LABELS).
   - Else: largest bright blob (Otsu).
   Then inverse-sqrt distance, bbox-split brightness, EMA smoothing.

TFLite object classifier branch in detect_laptop is temporarily commented out (Roboflow-only laptop gate).
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

import brightness as bright

# Laptop gate: re-enable alongside the TFLite block in detect_laptop().
# import object_inference as obj_inf

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


def _roboflow_annotated_image_always() -> bool:
    """When true, gated success responses include `annotated_image` even without include_laptop_details."""
    return os.environ.get("ROBOFLOW_RETURN_ANNOTATED_IMAGE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


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
    *,
    roboflow_detection_frame: np.ndarray | None = None,
) -> tuple[dict[str, Any], str, bool] | None:
    """
    Returns (det_dict, detection_source, detection_reliable), or None → use legacy distance path.

    Roboflow when gate is enabled and configured. (TFLite branch commented out temporarily.)
    Optional `roboflow_detection_frame` is sent to the workflow; `frame` is used for downstream analysis.
    """
    if rwf.gate_enabled() and rwf.configured():
        rf = roboflow_detection_frame if roboflow_detection_frame is not None else frame
        det = rwf.try_predict_for_gate(rf, width, height)
        if det is None:
            return None
        src = "roboflow"
        return det, src, _detection_reliable_for_classifier(det, src)

    # TFLite objects model (laptop gate): disabled — use Roboflow only. Restore import + block to re-enable.
    # if obj_inf.objects_model_loaded():
    #     det = obj_inf.predict_objects(frame)
    #     if det is None:
    #         return None
    #     src = "tflite"
    #     return det, src, _detection_reliable_for_classifier(det, src)

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


def _cref_from_dims(bw: int, bh: int) -> float:
    short_side = min(bw, bh)
    long_side = max(bw, bh)
    aspect = (long_side / short_side) if short_side > 0 else 0.0
    return 0.80 if 1.2 <= aspect <= 2.5 else 0.50


def _resolve_laptop_roi(
    frame: np.ndarray,
    width: int,
    height: int,
    det: dict[str, Any],
    detection_source: str,
) -> tuple[int, int, int, int, float, float, str] | None:
    """
    Pixel ROI for distance/brightness: Roboflow workflow bbox when present, else Otsu bright blob.

    Returns (x, y, bw, bh, bbox_area_ratio_pct, cref, bbox_source).
    """
    box = det.get("bbox")
    if (
        detection_source == "roboflow"
        and str(det.get("source", "")) == "roboflow_workflow"
        and isinstance(box, dict)
    ):
        try:
            x = int(box["x"])
            y = int(box["y"])
            bw = int(box["w"])
            bh = int(box["h"])
        except (KeyError, TypeError, ValueError):
            x = y = bw = bh = -1
        if x >= 0 and y >= 0 and bw > 0 and bh > 0:
            total_area = float(width * height)
            ratio_pct = (float(bw * bh) / total_area) * 100.0
            cref = _cref_from_dims(bw, bh)
            return x, y, bw, bh, ratio_pct, cref, "roboflow_workflow"

    blob = sd.find_largest_bright_bbox(frame, width, height)
    if blob is None:
        return None
    x, y, bw, bh, bbox_ratio, cref = blob
    return x, y, bw, bh, bbox_ratio, cref, "bright_blob"


def analyze_with_laptop_gate(
    frame: np.ndarray,
    width: int,
    height: int,
    *,
    include_laptop_details: bool = False,
    include_workflow_annotated_image: bool = False,
    roboflow_detection_frame: np.ndarray | None = None,
    cached_laptop_detection: tuple[dict[str, Any], str, bool] | None = None,
) -> dict[str, Any] | None:
    """
    Returns None if Roboflow gate is unavailable (objects TFLite laptop path is temporarily off).

    Otherwise returns a dict with detection_source / detection_reliable on every branch.

    `cached_laptop_detection` skips running Roboflow/TFLite (from a prior step_token flow).
    `roboflow_detection_frame` is only used when calling Roboflow; `frame` is always used
    for ROI/brightness/distance (latest glasses image).
    `include_workflow_annotated_image`: attach Roboflow `annotated_image` when present (e.g. /api/analyze).
    """
    if cached_laptop_detection is not None:
        det, detection_source, detection_reliable = cached_laptop_detection
    else:
        got = detect_laptop(
            frame,
            width,
            height,
            roboflow_detection_frame=roboflow_detection_frame,
        )
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

    roi = _resolve_laptop_roi(frame, width, height, det, detection_source)
    if roi is None:
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

    x, y, bw, bh, bbox_ratio, cref, bbox_source = roi
    wf_d = det.get("workflow_distance_cm")
    use_wf_dist = (
        detection_source == "roboflow"
        and str(det.get("source", "")) == "roboflow_workflow"
        and wf_d is not None
    )
    raw_d_wf: float | None = None
    if use_wf_dist:
        try:
            raw_d_wf = float(wf_d)
        except (TypeError, ValueError):
            raw_d_wf = None
    if use_wf_dist and raw_d_wf is not None:
        raw_d = raw_d_wf
        raw_d = float(np.clip(raw_d, sd.DISTANCE_CM_MIN, sd.DISTANCE_CM_MAX))
        raw_d = round(raw_d, 1)
        raw_r = float(bbox_ratio)
        sm_d, sm_r = tsmooth.apply_smoothing(raw_d, raw_r)
        dist_raw = {
            "distance_cm": sm_d,
            "category": sd.category_from_distance_cm(sm_d),
            "confidence": float(det.get("confidence", 0.0)),
            "source": "roboflow_workflow",
            "distance_source": "roboflow_workflow",
            "screen_ratio": int(round(float(np.clip(sm_r, 0.0, 100.0)))),
            "bbox_area_ratio_pct": sm_r,
        }
    else:
        dist_raw = sd.compute_distance(bbox_ratio, cref)
        raw_d = float(dist_raw["distance_cm"])
        raw_r = float(dist_raw.get("bbox_area_ratio_pct", bbox_ratio))
        sm_d, sm_r = tsmooth.apply_smoothing(raw_d, raw_r)
        dist_raw["distance_cm"] = sm_d
        dist_raw["category"] = sd.category_from_distance_cm(sm_d)
        dist_raw["screen_ratio"] = int(round(float(np.clip(sm_r, 0.0, 100.0))))
        dist_raw["bbox_area_ratio_pct"] = sm_r
        dist_raw["distance_source"] = dist_raw.get("source", "laptop_bbox")

    br = compute_brightness(frame, x, y, bw, bh, width, height)

    merged: dict[str, Any] = {
        "laptop_detected": True,
        "analysis_ready": True,
        "laptop_class_id": det["class_id"],
        "laptop_class": det["class_name"],
        "laptop_confidence": det["confidence"],
        "bbox": {"x": int(x), "y": int(y), "w": int(bw), "h": int(bh)},
        "bbox_source": bbox_source,
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
    ann = det.get("annotated_image")
    if isinstance(ann, str) and ann.strip() and (
        include_laptop_details
        or include_workflow_annotated_image
        or _roboflow_annotated_image_always()
    ):
        merged["annotated_image"] = ann.strip()
    return merged
