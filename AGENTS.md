## Learned User Preferences

- For apps using `python-dotenv`, keep `.env` in `KEY=value` form; PowerShell `$env:` assignment lines are not applied by `load_dotenv()`.
- Saved glasses PNGs default to per-frame contrast stretch for visibility; set `GLASSES_PNG_RAW=1` when PNG pixels must match raw sensor values. For OpenAI vision PNG encoding, see `OPENAI_PNG_RAW` in `openai_screen.py`.

## Learned Workspace Facts

- `POST /api/frame` (and `POST /api/analyze` plus path aliases) decodes the frame, stores the latest image for the background loop, and saves a PNG under `images/glasses/`. Saving is not gated on `COLLECT_GLASSES_FRAMES` (that env toggle was removed from `main.py`).
- Payload: base64 raw bytes of length `width*height` (grayscale) **or** `width*height*2` (RGB565 LE); see `frame_codec.py`. Roboflow path uses BGR; brightness / blob logic uses derived grayscale.
- Clients often poll `GET /api/latest-result` after posting a frame. `GET /api/analyze` is a lightweight tunnel ping; `GET /` lists main routes.
- Roboflow workflows are called via HTTP in `roboflow_workflow.py` so the project can run on Python 3.13 without the `inference-sdk` PyPI package.
- The Roboflow laptop gate is on when `ROBOFLOW_API_KEY`, `ROBOFLOW_WORKSPACE`, and `ROBOFLOW_WORKFLOW_ID` are set, unless `LAPTOP_USE_ROBOFLOW_WORKFLOW` explicitly opts out.
- In `laptop_scene.py`, the TFLite laptop-detection branch for the gate path is commented out (Roboflow-only for that path); other TFLite endpoints may remain for objects, etc.
- Glasses frames are saved under `images/glasses/`; `/api/collect` writes to `images/too_close`, `images/ok`, and `images/far`; `images/` is gitignored.
- Typical glasses frame from current firmware (`script.cpp`) is **80×60 grayscale** over BLE (**4800** bytes); **9600** bytes means RGB565 LE per `frame_codec.py`. Other `width`/`height` are allowed if the payload length matches.
- `POST /api/openai-frame` sends the frame image to OpenAI Vision (requires `OPENAI_API_KEY` or legacy `OPEN_AI_API_KEY`); returns screen vs environment brightness estimates and `distance_cm`. Server logs a one-line summary to the console.
- ESP32 firmware (`script.cpp`) streams frames over BLE to the phone only; it does not call the HTTP API. The mobile app must reassemble NOTIFY chunks (**4800-byte grayscale** or **9600-byte RGB565**, depending on firmware) and POST to the backend.
- For `*.ngrok-free.dev` / `*.ngrok-free.app`, non-browser HTTP clients usually need header `ngrok-skip-browser-warning` (any value) or a custom `User-Agent`; otherwise the ngrok interstitial HTML is returned instead of JSON.
