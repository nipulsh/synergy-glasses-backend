"""
Screen vs ambient brightness from an 80×60 grayscale frame (uint8, row-major).

- Screen: centre 50% (same relative box for any H×W).
- Ambient: darkest of four corner patches (bezel / wall / off-screen), not the
  full border ring — the ring often matches the LCD and kills contrast.
"""

from __future__ import annotations

import numpy as np

BrightnessAlertLiteral = str  # "OK" | "HIGH_CONTRAST" | "BOTH_BRIGHT"


def analyze_brightness(frame: np.ndarray, width: int, height: int) -> dict:
    """
    Args:
        frame: (H, W) uint8 grayscale.
        width, height: Expected dimensions (validated against frame.shape).

    Returns:
        {
            "screenBr": 0–100,
            "ambientBr": 0–100,
            "contrast": 0–100,
            "brightnessAlert": "OK" | "HIGH_CONTRAST" | "BOTH_BRIGHT",
        }
    """
    if frame.ndim != 2:
        raise ValueError("frame must be 2D (H, W)")

    h, w = int(frame.shape[0]), int(frame.shape[1])
    if h != height or w != width:
        raise ValueError(f"frame shape ({h}, {w}) does not match height={height}, width={width}")
    if h < 10 or w < 10:
        raise ValueError("frame too small for brightness ROIs")

    f = frame.astype(np.float64)

    # Centre 50%
    r0, r1 = h // 4, h - h // 4
    c0, c1 = w // 4, w - w // 4
    screen_roi = f[r0:r1, c0:c1]

    # Corner patches: more likely off-LCD than a thin border ring (which samples
    # the same panel and yields ~0 contrast vs centre).
    ch = max(3, min(h // 6, h // 2))
    cw = max(3, min(w // 6, w // 2))
    corner_blocks = (
        f[:ch, :cw],
        f[:ch, w - cw :],
        f[h - ch :, :cw],
        f[h - ch :, w - cw :],
    )
    corner_means = [float(np.mean(block)) for block in corner_blocks if block.size]
    ambient_sample = min(corner_means) if corner_means else 0.0

    def to_pct(mean_val: float) -> float:
        return round(float(np.clip(mean_val / 255.0 * 100.0, 0.0, 100.0)), 1)

    screen_br = to_pct(float(np.mean(screen_roi))) if screen_roi.size else 0.0
    ambient_br = to_pct(ambient_sample)
    contrast = round(abs(screen_br - ambient_br), 1)

    if screen_br >= 70.0 and ambient_br >= 70.0:
        alert: BrightnessAlertLiteral = "BOTH_BRIGHT"
    elif contrast >= 30.0:
        alert = "HIGH_CONTRAST"
    else:
        alert = "OK"

    return {
        "screenBr": screen_br,
        "ambientBr": ambient_br,
        "contrast": contrast,
        "brightnessAlert": alert,
    }
