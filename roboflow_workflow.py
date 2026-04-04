"""
Optional Roboflow Inference HTTP workflow for laptop gating (inference only — not training).

Requires `inference-sdk` (Python 3.9–3.12; see requirements-inference-sdk.txt) and env vars.
Used only when `objects_model.tflite` is missing but the Roboflow gate is enabled (not on the hot path when TFLite is loaded).

Training from a Roboflow **Classification** export: use `train_objects.py --source folder --data-dir …`.
"""

from __future__ import annotations

import json
import os
from typing import Any

import cv2
import numpy as np

_WORKFLOW_IMAGE_KEY_DEFAULT = "image"
_inference_client: Any = None


def gate_enabled() -> bool:
    return os.environ.get("LAPTOP_USE_ROBOFLOW_WORKFLOW", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def configured() -> bool:
    return bool(
        os.environ.get("ROBOFLOW_API_KEY", "").strip()
        and os.environ.get("ROBOFLOW_WORKSPACE", "").strip()
        and os.environ.get("ROBOFLOW_WORKFLOW_ID", "").strip()
    )


def _get_inference_client() -> Any:
    """Singleton HTTP client — avoids per-request construction."""
    global _inference_client
    if _inference_client is not None:
        return _inference_client
    from inference_sdk import InferenceHTTPClient  # noqa: PLC0415

    api_url = os.environ.get("ROBOFLOW_API_URL", "https://detect.roboflow.com").strip()
    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    _inference_client = InferenceHTTPClient(api_url=api_url, api_key=api_key)
    return _inference_client


def _workflow_parameters() -> dict[str, Any] | None:
    """
    Extra workflow inputs beyond `images`. Matches Roboflow snippets that omit parameters;
    set ROBOFLOW_WORKFLOW_PARAMETERS_JSON or ROBOFLOW_WORKFLOW_CLASSES when your workflow needs them.
    """
    raw = os.environ.get("ROBOFLOW_WORKFLOW_PARAMETERS_JSON", "").strip()
    if raw:
        return json.loads(raw)
    classes = os.environ.get("ROBOFLOW_WORKFLOW_CLASSES")
    if classes is not None and str(classes).strip():
        return {"classes": str(classes).strip()}
    return None


def _laptop_label_substrings() -> list[str]:
    return [
        x.strip().lower()
        for x in os.environ.get("ROBOFLOW_LAPTOP_LABELS", "laptop").split(",")
        if x.strip()
    ]


def _collect_label_conf_pairs(obj: Any, out: list[tuple[str, float]]) -> None:
    if isinstance(obj, dict):
        label: str | None = None
        for k in ("class", "class_name", "label", "name", "predicted_class", "top", "detection_class"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                label = v.strip()
                break
        conf: float | None = None
        for k in ("confidence", "score", "class_confidence", "probability", "conf"):
            if k not in obj:
                continue
            try:
                conf = float(obj[k])
                break
            except (TypeError, ValueError):
                pass
        if label is not None and conf is not None:
            out.append((label, conf))
        for v in obj.values():
            _collect_label_conf_pairs(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _collect_label_conf_pairs(v, out)


def _normalize_result(result: Any) -> Any:
    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"_unparsed": result}
    return result


def _unwrap_workflow_payload(result: Any) -> Any:
    """run_workflow returns a list of output dicts on newer SDKs."""
    if isinstance(result, list):
        return result[0] if result else {}
    return result


def detection_from_workflow_result(result: Any) -> dict[str, Any]:
    """
    Map arbitrary workflow JSON to the same shape as object_inference.predict_objects
    for laptop_scene._is_laptop (class_name, confidence, top3, source).
    """
    obj = _normalize_result(_unwrap_workflow_payload(result))
    pairs: list[tuple[str, float]] = []
    _collect_label_conf_pairs(obj, pairs)

    by_label: dict[str, float] = {}
    for lab, c in pairs:
        key = lab.strip()
        by_label[key] = max(by_label.get(key, 0.0), float(c))

    hints = _laptop_label_substrings()
    laptop_conf = 0.0
    laptop_name = ""
    for lab, c in by_label.items():
        low = lab.lower()
        if any(h in low for h in hints):
            if c > laptop_conf:
                laptop_conf = c
                laptop_name = lab

    top3 = sorted(by_label.items(), key=lambda x: -x[1])[:3]
    top3_out = [
        {"class_id": i, "class_name": name, "confidence": round(float(conf), 4)}
        for i, (name, conf) in enumerate(top3)
    ]

    return {
        "class_id": 0 if laptop_name else -1,
        "class_name": laptop_name or (top3[0][0] if top3 else ""),
        "confidence": round(float(laptop_conf), 4),
        "top3": top3_out,
        "source": "roboflow_workflow",
    }


def run_workflow_on_gray_frame(frame: np.ndarray, width: int, height: int) -> Any:
    """Send frame as in-memory BGR image (no temp files)."""
    if frame.ndim != 2:
        raise ValueError("frame must be 2D (H, W)")
    h, w = int(frame.shape[0]), int(frame.shape[1])
    if h != height or w != width:
        raise ValueError(f"frame shape ({h}, {w}) does not match height={height}, width={width}")

    workspace = os.environ.get("ROBOFLOW_WORKSPACE", "").strip()
    workflow_id = os.environ.get("ROBOFLOW_WORKFLOW_ID", "").strip()
    image_key = os.environ.get("ROBOFLOW_WORKFLOW_IMAGE_KEY", _WORKFLOW_IMAGE_KEY_DEFAULT).strip()

    client = _get_inference_client()
    bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    use_cache = (
        os.environ.get("ROBOFLOW_WORKFLOW_USE_CACHE", "true").strip().lower()
        not in ("0", "false", "no")
    )
    return client.run_workflow(
        workspace_name=workspace,
        workflow_id=workflow_id,
        images={image_key: bgr},
        parameters=_workflow_parameters(),
        use_cache=use_cache,
    )


def try_predict_for_gate(frame: np.ndarray, width: int, height: int) -> dict[str, Any] | None:
    """
    If gate is enabled and configured, run workflow and return a predict_objects-shaped dict.
    Returns None if disabled, misconfigured, or inference_sdk / network error.
    """
    if not gate_enabled() or not configured():
        return None
    try:
        raw = run_workflow_on_gray_frame(frame, width, height)
        return detection_from_workflow_result(raw)
    except Exception as e:
        print(f"[roboflow_workflow] workflow failed: {e}")
        return None
