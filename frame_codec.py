"""Decode glasses payloads (grayscale uint8 or RGB565 LE) and normalize for CV/ML."""

from __future__ import annotations

import cv2
import numpy as np


def _canonical_wh(raw_len: int, width: int, height: int) -> tuple[int, int]:
    """
    Row-major buffers from the glasses pipeline are 80×60 (QQVGA sub-sample).
    Clients often pass JSON width/height swapped as 60×80; byte count still matches
    (4800 gray or 9600 RGB565) but reshape then shears rows horizontally.
    """
    w, h = int(width), int(height)
    if (w, h) == (60, 80) and raw_len in (4800, 9600):
        return 80, 60
    return w, h


def rgb565_u16_to_bgr(u16_hw: np.ndarray) -> np.ndarray:
    """RGB565 little-endian pixels (h, w) uint16 → BGR uint8 (h, w, 3)."""
    x = u16_hw.astype(np.uint32)
    r5 = (x >> 11) & 0x1F
    g6 = (x >> 5) & 0x3F
    b5 = x & 0x1F
    r = ((r5 * 255 + 15) // 31).astype(np.uint8)
    g = ((g6 * 255 + 31) // 63).astype(np.uint8)
    b = ((b5 * 255 + 15) // 31).astype(np.uint8)
    return np.stack([b, g, r], axis=2)


def decode_frame_bytes(
    raw: bytes, width: int, height: int
) -> tuple[np.ndarray, str, int, int]:
    """
    Returns (array, pixel_format, width_used, height_used).

    - gray: shape (h, w) uint8, one byte per pixel (legacy).
    - rgb565: shape (h, w, 3) uint8 BGR (decoded from RGB565 LE).
    """
    n = len(raw)
    w, h = _canonical_wh(n, width, height)
    gray_n = w * h
    rgb565_n = w * h * 2
    if n == gray_n:
        arr = np.frombuffer(raw, dtype=np.uint8, count=gray_n).copy().reshape(h, w)
        return arr, "gray", w, h
    if n == rgb565_n:
        u16 = np.frombuffer(raw, dtype="<u2", count=w * h).reshape(h, w).copy()
        return rgb565_u16_to_bgr(u16), "rgb565", w, h
    raise ValueError(
        f"payload length {n} not w*h ({gray_n}) or w*h*2 RGB565 ({rgb565_n})"
    )


def as_gray(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Luminance for brightness / blobs; supports 1D gray, (h,w) gray, or (h,w,3) BGR."""
    w, h = int(width), int(height)
    if frame.ndim == 1:
        if frame.size != w * h:
            raise ValueError(f"flat gray size {frame.size} != {w * h}")
        return frame.reshape(h, w)
    if frame.ndim == 2:
        if frame.shape != (h, w):
            raise ValueError(f"gray shape {frame.shape} != ({h}, {w})")
        return np.ascontiguousarray(frame, dtype=np.uint8)
    if frame.ndim == 3 and frame.shape[2] == 3:
        if frame.shape[0] != h or frame.shape[1] != w:
            raise ValueError(f"BGR shape {frame.shape} != ({h}, {w}, 3)")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"unsupported frame shape {frame.shape}")


def as_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """BGR for Roboflow / JPEG encode; supports flat or 2D gray or BGR."""
    w, h = int(width), int(height)
    if frame.ndim == 1:
        g = frame.reshape(h, w) if frame.size == w * h else None
        if g is None:
            raise ValueError(f"flat gray size {frame.size} != {w * h}")
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 2:
        if frame.shape != (h, w):
            raise ValueError(f"gray shape {frame.shape} != ({h}, {w})")
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 3:
        if frame.shape[0] != h or frame.shape[1] != w:
            raise ValueError(f"BGR shape {frame.shape} != ({h}, {w}, 3)")
        return np.ascontiguousarray(frame, dtype=np.uint8)
    raise ValueError(f"unsupported frame shape {frame.shape}")
