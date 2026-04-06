"""Screen vs ambient brightness inside/outside laptop bbox."""

import numpy as np

import frame_codec


def _mean_uint8_to_pct(mean_val: float) -> float:
    return float(mean_val / 255.0 * 100.0)


def analyze_brightness_bbox(
    frame: np.ndarray, x: int, y: int, bw: int, bh: int, width: int, height: int
) -> dict:
    gray = frame_codec.as_gray(frame, width, height)
    h, w = gray.shape[:2]
    x0 = max(0, min(x, w - 1))
    y0 = max(0, min(y, h - 1))
    x1 = max(x0 + 1, min(x + bw, w))
    y1 = max(y0 + 1, min(y + bh, h))
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    total = float(h * w)
    inside = gray[mask == 1]
    outside = gray[mask == 0]
    screen_mean = float(np.mean(inside)) if inside.size else 0.0
    outside_frac = outside.size / total if total > 0 else 0.0
    if outside_frac < 0.05:
        ch, cw = max(1, h // 6), max(1, w // 6)
        corners = [
            gray[0:ch, 0:cw],
            gray[0:ch, w - cw : w],
            gray[h - ch : h, 0:cw],
            gray[h - ch : h, w - cw : w],
        ]
        corner_means = [float(np.mean(c)) for c in corners if c.size]
        ambient_mean = min(corner_means) if corner_means else 0.0
    else:
        ambient_mean = float(np.mean(outside)) if outside.size else 0.0
    screen_br = _mean_uint8_to_pct(screen_mean)
    ambient_br = _mean_uint8_to_pct(ambient_mean)
    contrast = abs(screen_br - ambient_br)
    if contrast < 10.0:
        alert = "DULL"
    elif contrast > 25.0:
        alert = "BRIGHT"
    else:
        alert = "MODERATE"
    return {
        "screenBr": round(screen_br, 2),
        "ambientBr": round(ambient_br, 2),
        "contrast": round(contrast, 2),
        "brightnessAlert": alert,
    }
