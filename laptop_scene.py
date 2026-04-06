"""Laptop detection gate, ROI selection, distance and brightness fusion."""

import os

import numpy as np

import brightness
import frame_codec
import roboflow_workflow
import screen_distance
import temporal_smoothing

LAPTOP_CONFIDENCE_MIN = float(os.environ.get("LAPTOP_CONFIDENCE_MIN", "0.4"))


def _is_laptop(det: dict) -> bool:
    return float(det.get("confidence", 0.0)) >= LAPTOP_CONFIDENCE_MIN


def _cref_from_aspect(bw: int, bh: int) -> float:
    if bh <= 0:
        return 0.50
    ar = bw / float(bh)
    if 1.2 <= ar <= 2.5:
        return 0.80
    return 0.50


def _resolve_laptop_roi(
    frame: np.ndarray, width: int, height: int, det: dict
) -> tuple[int, int, int, int, float, float, str] | None:
    gray = frame_codec.as_gray(frame, width, height)
    bbox = det.get("bbox")
    source = det.get("source", "")
    if isinstance(bbox, dict) and source == "roboflow_workflow":
        x = int(bbox["x"])
        y = int(bbox["y"])
        bw = int(bbox["w"])
        bh = int(bbox["h"])
        bbox_area_ratio_pct = 100.0 * (bw * bh) / float(width * height)
        cref = _cref_from_aspect(bw, bh)
        return x, y, bw, bh, bbox_area_ratio_pct, cref, "roboflow_workflow"
    blob = screen_distance.find_largest_bright_bbox(gray, width, height)
    if blob is None:
        return None
    x, y, bw, bh, bbox_area_ratio_pct, cref = blob
    return x, y, bw, bh, bbox_area_ratio_pct, cref, "bright_blob"


def analyze_with_laptop_gate(
    frame: np.ndarray,
    width: int,
    height: int,
    *,
    cached_laptop_detection: tuple[dict, str, bool] | None = None,
) -> dict | None:
    if cached_laptop_detection is None:
        det = roboflow_workflow.try_predict_for_gate(frame, width, height)
        if det is None:
            return None
        detection_source = "roboflow"
        detection_reliable = True
    else:
        det, detection_source, detection_reliable = cached_laptop_detection

    base_meta = {
        "laptop_class": str(det.get("class_name", "")),
        "laptop_confidence": float(det.get("confidence", 0.0)),
        "detection_source": detection_source,
        "detection_reliable": detection_reliable,
    }

    if not _is_laptop(det):
        return {
            "laptop_detected": False,
            "analysis_ready": False,
            **base_meta,
        }

    roi = _resolve_laptop_roi(frame, width, height, det)
    if roi is None:
        return {
            "laptop_detected": True,
            "analysis_ready": False,
            "reason": "no_screen_blob",
            **base_meta,
        }

    x, y, bw, bh, bbox_area_ratio_pct, cref, bbox_source = roi
    dist_info = screen_distance.compute_distance(bbox_area_ratio_pct, cref)
    raw_d = float(dist_info["distance_cm"])
    raw_r = float(dist_info["bbox_area_ratio_pct"])
    smoothed_d, smoothed_r = temporal_smoothing.apply_smoothing(raw_d, raw_r)
    category = screen_distance.category_from_distance_cm(smoothed_d)
    screen_ratio = int(round(smoothed_r))

    br = brightness.analyze_brightness_bbox(frame, x, y, bw, bh, width, height)

    return {
        "laptop_detected": True,
        "analysis_ready": True,
        "laptop_class": str(det.get("class_name", "")),
        "laptop_confidence": float(det.get("confidence", 0.0)),
        "bbox": {"x": x, "y": y, "w": bw, "h": bh},
        "bbox_source": bbox_source,
        "distance_cm": smoothed_d,
        "category": category,
        "screen_ratio": screen_ratio,
        "screenBr": br["screenBr"],
        "ambientBr": br["ambientBr"],
        "contrast": br["contrast"],
        "brightnessAlert": br["brightnessAlert"],
        "detection_source": detection_source,
        "detection_reliable": detection_reliable,
    }
