#!/usr/bin/env python3
"""Check /api/health (OpenAI key seen by server) and optionally POST /api/openai-frame.

Usage:
  python scripts/verify_openai_and_health.py http://127.0.0.1:8000
  python scripts/verify_openai_and_health.py https://YOUR.ngrok-free.dev --ngrok

Requires OPENAI_API_KEY (or OPEN_AI_API_KEY) in synergy-glasses-backend/.env next to main.py.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.error
import urllib.request


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base_url", help="e.g. http://127.0.0.1:8000 or https://xxx.ngrok-free.dev")
    p.add_argument("--ngrok", action="store_true", help="send ngrok-skip-browser-warning: 1")
    p.add_argument(
        "--skip-vision",
        action="store_true",
        help="only GET /api/health (no OpenAI call, no charge)",
    )
    args = p.parse_args()

    root = args.base_url.rstrip("/")
    headers = {"Content-Type": "application/json"}
    if args.ngrok:
        headers["ngrok-skip-browser-warning"] = "1"

    # Health
    try:
        req = urllib.request.Request(f"{root}/api/health", headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=15) as resp:
            health = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print("GET /api/health failed:", e.code, e.read().decode(errors="replace")[:500], file=sys.stderr)
        return 1
    except urllib.error.URLError as e:
        print("GET /api/health:", e, file=sys.stderr)
        return 1

    print("GET /api/health:", json.dumps(health, indent=2))
    if not health.get("openai_configured"):
        print(
            "\nopenai_configured is false — set OPENAI_API_KEY in .env next to main.py and restart uvicorn.",
            file=sys.stderr,
        )
        return 2

    if args.skip_vision:
        print("\n--skip-vision: not calling OpenAI.")
        return 0

    # Minimal valid grayscale frame (4800 bytes) — OpenAI still gets a real PNG of a flat image.
    w, h = 80, 60
    raw = bytes(w * h)  # black frame
    body = json.dumps(
        {"frame": base64.b64encode(raw).decode("ascii"), "width": w, "height": h}
    ).encode("utf-8")

    try:
        req = urllib.request.Request(
            f"{root}/api/openai-frame",
            data=body,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            out = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print("POST /api/openai-frame failed:", e.code, e.read().decode(errors="replace")[:2000], file=sys.stderr)
        return 1
    except urllib.error.URLError as e:
        print("POST /api/openai-frame:", e, file=sys.stderr)
        return 1

    print("\nPOST /api/openai-frame (OpenAI vision on PNG derived from frame):", json.dumps(out, indent=2))
    if "distance_cm" in out:
        print("\nOK: OpenAI returned JSON with distance_cm — images are reaching the model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
