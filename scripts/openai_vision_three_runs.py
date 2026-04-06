#!/usr/bin/env python3
"""Run OpenAI vision analysis on the 3 newest PNGs under images/glasses/ (test harness)."""

from __future__ import annotations

import sys
from pathlib import Path

# Repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import openai_screen


def main() -> int:
    glasses = ROOT / "images" / "glasses"
    if not glasses.is_dir():
        print(f"No folder {glasses}", file=sys.stderr)
        return 1

    pngs = sorted(
        glasses.glob("*.png"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:3]
    if len(pngs) < 1:
        print(f"No PNG files in {glasses}", file=sys.stderr)
        return 1

    if openai_screen.api_key() is None:
        print(
            "Set OPENAI_API_KEY in .env as a single line: OPENAI_API_KEY=sk-...\n"
            "(python-dotenv does not load PowerShell $env: lines.)",
            file=sys.stderr,
        )
        return 1

    print(f"Running {len(pngs)} OpenAI vision test(s)…\n", flush=True)
    for i, p in enumerate(pngs, 1):
        print(f"--- Run {i}/3: {p.name} ---", flush=True)
        try:
            out = openai_screen.analyze_png_path(str(p))
        except Exception as e:
            print(f"ERROR: {e}", flush=True)
            return 1
        print(f"  screen_brightness_0_100:    {out['screen_brightness_0_100']}", flush=True)
        print(f"  environment_brightness_0_100: {out['environment_brightness_0_100']}", flush=True)
        print(f"  brightness_difference_0_100: {out['brightness_difference_0_100']}", flush=True)
        print(f"  distance_cm:                {out['distance_cm']}", flush=True)
        print(f"  openai_confidence:          {out['openai_confidence']}", flush=True)
        print(f"  brief_notes:                {out.get('brief_notes', '')}", flush=True)
        print(flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
