"""Roboflow workflow client for laptop gating (HTTP JSON API; no inference-sdk required)."""

import base64
import json
import os
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import quote

import cv2
import numpy as np

import frame_codec

_LABEL_KEYS = ("class", "class_name", "label", "predicted_class", "top", "detection_class")
_CONF_KEYS = ("confidence", "score", "probability")


def gate_enabled() -> bool:
    v = os.environ.get("LAPTOP_USE_ROBOFLOW_WORKFLOW", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def configured() -> bool:
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    ws = os.environ.get("ROBOFLOW_WORKSPACE", "").strip()
    wid = os.environ.get("ROBOFLOW_WORKFLOW_ID", "").strip()
    return bool(key and ws and wid)


def _numpy_bgr_to_base64_jpeg(bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", bgr)
    if not ok:
        raise ValueError("cv2.imencode failed for JPEG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _run_workflow_http(
    bgr: np.ndarray,
    *,
    api_url: str,
    api_key: str,
    workspace: str,
    workflow_id: str,
    image_key: str,
) -> Any:
    """Same contract as inference_sdk InferenceHTTPClient.run_workflow (named workflow)."""
    base = api_url.rstrip("/")
    url = f"{base}/{quote(workspace, safe='')}/workflows/{quote(workflow_id, safe='')}"
    payload = {
        "api_key": api_key,
        "use_cache": True,
        "enable_profiling": False,
        "inputs": {
            image_key: {"type": "base64", "value": _numpy_bgr_to_base64_jpeg(bgr)},
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        print(f"Roboflow workflow HTTP {e.code}: {err_body[:500]}")
        raise
    outputs = raw.get("outputs")
    return outputs if outputs is not None else raw


def _laptop_hints() -> list[str]:
    raw = os.environ.get("ROBOFLOW_LAPTOP_LABELS", "laptop").strip().lower()
    return [p.strip() for p in raw.split(",") if p.strip()]


def _label_matches_laptop(label: str) -> bool:
    if not label:
        return False
    low = label.lower()
    for hint in _laptop_hints():
        if hint in low:
            return True
    return False


def _iter_nodes(obj: Any) -> Any:
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _iter_nodes(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_nodes(item)


def _get_label_value(d: dict) -> str | None:
    for k in _LABEL_KEYS:
        if k in d and d[k] is not None:
            v = d[k]
            if isinstance(v, str):
                return v
            if isinstance(v, (int, float)):
                return str(v)
    return None


def _get_conf_value(d: dict) -> float | None:
    for k in _CONF_KEYS:
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except (TypeError, ValueError):
                continue
    return None


def _is_probably_normalized(vals: list[float]) -> bool:
    if not vals:
        return False
    return all(0.0 <= v <= 1.0 for v in vals)


def _extract_bbox_from_dict(
    d: dict, frame_w: int, frame_h: int
) -> dict[str, int] | None:
    wk = None
    hk = None
    width = d.get("width", d.get("w"))
    height = d.get("height", d.get("h"))
    if width is None or height is None:
        return None
    try:
        fw = float(width)
        fh = float(height)
    except (TypeError, ValueError):
        return None

    x_min = d.get("x_min", d.get("xmin", d.get("left")))
    y_min = d.get("y_min", d.get("ymin", d.get("top")))
    x_max = d.get("x_max", d.get("xmax", d.get("right")))
    y_max = d.get("y_max", d.get("ymax", d.get("bottom")))

    cx = d.get("x", d.get("center_x", d.get("cx")))
    cy = d.get("y", d.get("center_y", d.get("cy")))

    norm = False
    if x_min is not None and y_min is not None and x_max is not None and y_max is not None:
        try:
            xa, ya, xb, yb = float(x_min), float(y_min), float(x_max), float(y_max)
        except (TypeError, ValueError):
            return None
        norm = _is_probably_normalized([xa, ya, xb, yb])
        if norm:
            xa, xb = xa * frame_w, xb * frame_w
            ya, yb = ya * frame_h, yb * frame_h
        x0 = int(max(0, min(round(xa), frame_w - 1)))
        y0 = int(max(0, min(round(ya), frame_h - 1)))
        x1 = int(max(x0 + 1, min(round(xb), frame_w)))
        y1 = int(max(y0 + 1, min(round(yb), frame_h)))
        bw = max(1, x1 - x0)
        bh = max(1, y1 - y0)
        return {"x": x0, "y": y0, "w": bw, "h": bh}

    if cx is None or cy is None:
        return None
    try:
        cx_f = float(cx)
        cy_f = float(cy)
    except (TypeError, ValueError):
        return None
    norm = _is_probably_normalized([cx_f, cy_f, fw, fh])
    if norm:
        cx_f *= frame_w
        cy_f *= frame_h
        fw *= frame_w
        fh *= frame_h
    half_w = fw / 2.0
    half_h = fh / 2.0
    x0 = int(round(cx_f - half_w))
    y0 = int(round(cy_f - half_h))
    x0 = max(0, min(x0, frame_w - 1))
    y0 = max(0, min(y0, frame_h - 1))
    bw = max(1, int(round(fw)))
    bh = max(1, int(round(fh)))
    x1 = min(frame_w, x0 + bw)
    y1 = min(frame_h, y0 + bh)
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    return {"x": x0, "y": y0, "w": bw, "h": bh}


def detection_from_workflow_result(result: Any, frame_w: int, frame_h: int) -> dict:
    pairs: list[tuple[str, float, dict | None]] = []
    for d in _iter_nodes(result):
        label = _get_label_value(d)
        conf = _get_conf_value(d)
        if label is None or conf is None:
            continue
        bb = _extract_bbox_from_dict(d, frame_w, frame_h)
        pairs.append((label, float(conf), bb))

    laptop_candidates = [(lab, c, b) for lab, c, b in pairs if _label_matches_laptop(lab)]
    other_candidates = [(lab, c, b) for lab, c, b in pairs if not _label_matches_laptop(lab)]

    chosen: tuple[str, float, dict | None] | None = None
    if laptop_candidates:
        laptop_candidates.sort(key=lambda t: t[1], reverse=True)
        chosen = laptop_candidates[0]
    elif pairs:
        pairs.sort(key=lambda t: t[1], reverse=True)
        chosen = pairs[0]

    top3_sorted = sorted(pairs, key=lambda t: t[1], reverse=True)[:3]
    top3 = [{"label": lab, "confidence": round(c, 4)} for lab, c, _ in top3_sorted]

    if chosen is None:
        return {
            "class_id": -1,
            "class_name": "",
            "confidence": 0.0,
            "top3": top3,
            "source": "roboflow_workflow",
        }

    lab, conf, bb = chosen
    if _label_matches_laptop(lab):
        out: dict = {
            "class_id": 0,
            "class_name": lab,
            "confidence": round(float(conf), 4),
            "top3": top3,
            "source": "roboflow_workflow",
        }
        if bb is not None:
            out["bbox"] = bb
        return out

    return {
        "class_id": -1,
        "class_name": lab,
        "confidence": round(float(conf), 4),
        "top3": top3,
        "source": "roboflow_workflow",
    }


def try_predict_for_gate(frame: np.ndarray, width: int, height: int) -> dict | None:
    if not gate_enabled() or not configured():
        return None
    try:
        api_url = os.environ.get("ROBOFLOW_API_URL", "https://detect.roboflow.com").rstrip("/")
        api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
        workspace = os.environ.get("ROBOFLOW_WORKSPACE", "").strip()
        workflow_id = os.environ.get("ROBOFLOW_WORKFLOW_ID", "").strip()
        image_key = os.environ.get("ROBOFLOW_WORKFLOW_IMAGE_KEY", "image").strip() or "image"
        bgr = frame_codec.as_bgr(frame, width, height)
        raw = _run_workflow_http(
            bgr,
            api_url=api_url,
            api_key=api_key,
            workspace=workspace,
            workflow_id=workflow_id,
            image_key=image_key,
        )
        return detection_from_workflow_result(raw, width, height)
    except Exception as e:
        print(f"Roboflow workflow error: {e}")
        return None
