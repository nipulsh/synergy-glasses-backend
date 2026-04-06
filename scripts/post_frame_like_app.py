#!/usr/bin/env python3
"""POST one RGB565 80x60 frame like the mobile app should (local or ngrok URL)."""

from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.error
import urllib.request


def _ngrok_offline_help() -> None:
    print(
        "\nERR_NGROK_3200 / endpoint offline means ngrok has no live tunnel for that hostname.\n"
        "  1) In another terminal run: ngrok http 8000\n"
        "  2) Copy the **Forwarding** URL exactly, e.g. https://blessedly-electroscopic-quintin.ngrok-free.dev\n"
        "     (do not use the literal text YOUR-SUBDOMAIN or xxxx — use your real random subdomain.)\n"
        "  3) Keep that ngrok process running while you test; free URLs change each time you restart ngrok.\n"
        "  4) Run uvicorn on port 8000 on the same PC: uvicorn main:app --host 0.0.0.0 --port 8000\n",
        file=sys.stderr,
    )


def _looks_like_placeholder_base_url(u: str) -> bool:
    low = u.lower()
    return "your-subdomain" in low or ("xxxx" in low and "ngrok" in low)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "base_url",
        help="Scheme + host only, e.g. http://127.0.0.1:8000 or https://<random>.ngrok-free.dev from ngrok UI",
    )
    p.add_argument(
        "--path",
        default="/api/analyze",
        help="POST path (default /api/analyze)",
    )
    p.add_argument("--ngrok", action="store_true", help="add ngrok-skip-browser-warning")
    args = p.parse_args()

    if _looks_like_placeholder_base_url(args.base_url):
        print(
            "base_url looks like a placeholder. Use the real https host from the ngrok window.",
            file=sys.stderr,
        )
        _ngrok_offline_help()
        return 2

    w, h = 80, 60
    raw = bytes(w * h * 2)  # dummy RGB565; replace with real payload for tests
    body = {
        "frame": base64.b64encode(raw).decode("ascii"),
        "width": w,
        "height": h,
    }
    url = args.base_url.rstrip("/") + args.path
    data = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if args.ngrok:
        headers["ngrok-skip-browser-warning"] = "1"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            print(resp.status, resp.read().decode()[:500])
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:1200]
        print(e.code, body, file=sys.stderr)
        if "ERR_NGROK_3200" in body or "offline" in body.lower():
            _ngrok_offline_help()
        return 1
    except urllib.error.URLError as e:
        print(e, file=sys.stderr)
        _ngrok_offline_help()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
