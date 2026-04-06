"""Send glasses frame to OpenAI Vision; brightness (screen vs environment) and distance.

Uses the official openai SDK with built-in retry / rate-limit back-off.
"""

from __future__ import annotations

import base64
import json
import os
import time

import cv2
import numpy as np
from openai import OpenAI, APIConnectionError, RateLimitError, APIStatusError

import frame_codec

_client: OpenAI | None = None

MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 2.0

# Env var names checked in order (set one in `.env` as KEY=value; see `.env.example`).
OPENAI_API_KEY_ENV_VARS = ("OPENAI_API_KEY", "OPEN_AI_API_KEY")


def api_key() -> str | None:
    for name in OPENAI_API_KEY_ENV_VARS:
        k = os.environ.get(name)
        if k and k.strip():
            return k.strip()
    return None


def _get_client() -> OpenAI:
    global _client
    key = api_key()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set (or legacy OPEN_AI_API_KEY)")
    if _client is None:
        _client = OpenAI(api_key=key, max_retries=MAX_RETRIES, timeout=90.0)
    return _client


def vision_model() -> str:
    return (os.environ.get("OPENAI_VISION_MODEL") or "gpt-4o").strip()


def _frame_to_png_b64(frame: np.ndarray, width: int, height: int) -> str:
    """Encode frame as PNG base64. Uses BGR if available, else stretched grayscale."""
    w, h = int(width), int(height)
    raw_png = (os.environ.get("OPENAI_PNG_RAW") or "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    if frame.ndim == 3 and frame.shape == (h, w, 3):
        bgr = np.ascontiguousarray(frame, dtype=np.uint8)
    else:
        gray = frame_codec.as_gray(frame, w, h)
        if not raw_png:
            g = gray.astype(np.float32)
            lo, hi = float(g.min()), float(g.max())
            if hi > lo:
                gray = ((g - lo) * (255.0 / (hi - lo))).clip(0, 255).astype(np.uint8)
            else:
                gray = np.ascontiguousarray(gray, dtype=np.uint8)
        else:
            gray = np.ascontiguousarray(gray, dtype=np.uint8)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    ok, buf = cv2.imencode(".png", bgr)
    if not ok or buf is None:
        raise ValueError("failed to encode PNG")
    return base64.standard_b64encode(buf.tobytes()).decode("ascii")


_SYSTEM = """You analyze one small image from smart glasses (often facing a laptop or desk).
Respond with a single JSON object only (no markdown).

Required keys:
- "screen_brightness_0_100": number 0-100, perceived brightness of the laptop/screen region (or brightest screen-like rectangle if visible).
- "environment_brightness_0_100": number 0-100, perceived brightness of the surrounding environment (desk, wall, room) excluding the screen rectangle as much as possible.
- "brightness_difference_0_100": number 0-100, absolute value |screen - environment| (contrast between screen and ambient).
- "distance_cm": number, your best estimate of distance from the camera (glasses) to the screen in centimeters; use range 20-150; if uncertain, give a mid estimate and lower confidence.
- "confidence": number 0-1, how reliable you consider screen detection and the estimates.
- "brief_notes": string, under 120 characters, what you see (e.g. "dark room, bright monitor").

If no clear screen, still give best-effort numbers and confidence under 0.35."""


def analyze_frame(frame: np.ndarray, width: int, height: int) -> dict:
    client = _get_client()
    model = vision_model()

    png_b64 = _frame_to_png_b64(frame, width, height)
    data_url = f"data:image/png;base64,{png_b64}"

    user_text = (
        "From this glasses-camera frame, estimate:\n"
        "1) screen vs environment brightness (each 0-100) and their absolute difference;\n"
        "2) distance from camera to screen in cm.\n"
        "Return only the JSON object specified in your instructions."
    )

    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=500,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_text},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    },
                ],
            )
            break
        except RateLimitError as e:
            last_err = e
            wait = RETRY_BACKOFF_SEC * (2 ** attempt)
            print(f"[openai] rate-limited, retrying in {wait:.1f}s (attempt {attempt+1})", flush=True)
            time.sleep(wait)
        except APIConnectionError as e:
            last_err = e
            wait = RETRY_BACKOFF_SEC * (2 ** attempt)
            print(f"[openai] connection error, retrying in {wait:.1f}s (attempt {attempt+1})", flush=True)
            time.sleep(wait)
        except APIStatusError as e:
            raise RuntimeError(f"OpenAI API error {e.status_code}: {e.message}") from e
    else:
        raise RuntimeError(f"OpenAI failed after {MAX_RETRIES+1} attempts: {last_err}")

    choice = resp.choices[0] if resp.choices else None
    if not choice or not choice.message or not choice.message.content:
        raise RuntimeError("OpenAI response missing message content")

    content = choice.message.content
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"OpenAI returned non-JSON: {content[:200]}") from e

    def _f(key: str, default: float = 0.0) -> float:
        try:
            return float(parsed.get(key, default))
        except (TypeError, ValueError):
            return default

    screen_b = _f("screen_brightness_0_100")
    env_b = _f("environment_brightness_0_100")
    diff = _f("brightness_difference_0_100")
    if diff <= 0 and (screen_b > 0 or env_b > 0):
        diff = abs(screen_b - env_b)
    dist = _f("distance_cm")
    conf = _f("confidence")
    notes = parsed.get("brief_notes")
    if not isinstance(notes, str):
        notes = ""

    return {
        "screen_brightness_0_100": round(screen_b, 2),
        "environment_brightness_0_100": round(env_b, 2),
        "brightness_difference_0_100": round(diff, 2),
        "distance_cm": round(dist, 2),
        "openai_confidence": round(conf, 3),
        "brief_notes": notes[:200],
        "openai_model": model,
        "screen_environment_diff_pct": round(diff, 2),
    }


def analyze_png_path(path: str) -> dict:
    """Load a PNG from disk (BGR) and run vision analysis. Uses image width/height from file."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cannot read image: {path}")
    h, w = bgr.shape[:2]
    return analyze_frame(bgr, w, h)
