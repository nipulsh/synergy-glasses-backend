Build a Python FastAPI backend called "OcuSmart Distance API" from scratch.
This is a real-time eye-health monitoring backend for smart glasses.

---

## What it does

Smart glasses (ESP32-S3 with OV2640 camera) stream grayscale frames to this
server. The server continuously sends those frames to Roboflow to detect
whether the user is looking at a laptop screen, estimates how far they are
from the screen, measures screen brightness vs ambient light, and makes
the result available for a mobile app to poll.

---

## Data flow (critical — build around this)

1. Glasses POST a frame every ~2 seconds to `POST /api/frame`.
   Backend stores it as the "latest frame" in memory. Returns immediately.

2. A background loop runs forever on the server:
   a. Take the latest stored frame.
   b. Send it to Roboflow. Wait however long it takes (no timeout).
   c. Once Roboflow responds, take the LATEST frame in memory at that
   moment (may be newer than what was sent to Roboflow).
   d. Use the Roboflow detection result + that latest frame to compute
   distance and brightness.
   e. Store the result as "latest result".
   f. Immediately loop back to step (a). No sleep between iterations.

3. The mobile app polls `GET /api/latest-result` every ~2 seconds to
   display distance and brightness alerts to the user.

---

## Project structure

```
backend/
  main.py                  # FastAPI app, frame buffer, background loop, routes
  roboflow_workflow.py     # Roboflow HTTP client
  laptop_scene.py          # laptop detection gate + ROI selection
  screen_distance.py       # distance from bbox fill ratio
  brightness.py            # screen vs ambient brightness split
  temporal_smoothing.py    # EMA smoothing on distance + ratio
  collector.py             # save labeled frames for training dataset
  requirements.txt
  .env.example
  models/                  # TFLite models placed here (optional)
```

---

## requirements.txt

```
fastapi>=0.115
uvicorn[standard]>=0.30
python-dotenv>=1.0
numpy>=1.26
opencv-python-headless>=4.9
pydantic>=2.6
python-multipart>=0.0.9
# inference-sdk: install separately in a Python 3.12 venv
# pip install inference-sdk
```

---

## .env.example

```
LAPTOP_USE_ROBOFLOW_WORKFLOW=1
ROBOFLOW_API_KEY=
ROBOFLOW_WORKSPACE=
ROBOFLOW_WORKFLOW_ID=
ROBOFLOW_API_URL=https://detect.roboflow.com
ROBOFLOW_WORKFLOW_IMAGE_KEY=image
ROBOFLOW_LAPTOP_LABELS=laptop
LAPTOP_CONFIDENCE_MIN=0.4
DISTANCE_D_REF=60.0
DISTANCE_REF_RATIO_PCT=25.0
DISTANCE_MIN_RATIO_PCT=0.5
DISTANCE_CM_MIN=20.0
DISTANCE_CM_MAX=150.0
CLIENT_DATA_POLL_INTERVAL_SEC=2
CLIENT_ALERT_COOLDOWN_SEC=20
```

---

## Frame format (all POST endpoints that accept a frame)

```json
{
  "frame": "<base64-encoded raw grayscale bytes, width × height uint8>",
  "width": 80,
  "height": 60
}
```

Validate: `len(base64_decoded) == width * height`. Reshape to `(height, width)` numpy uint8.

---

## main.py

Load `.env` before any module import (use `python-dotenv`).

### In-memory state (module-level, thread-safe with threading.Lock)

```python
_latest_frame: np.ndarray | None = None
_latest_frame_meta: dict | None = None   # {"width": int, "height": int}
_frame_lock = threading.Lock()

_latest_result: dict | None = None
_result_lock = threading.Lock()
```

### Background loop

Start as an asyncio task in the FastAPI lifespan (use `@asynccontextmanager`
and `asyncio.create_task`). The loop body must run in a thread executor
(`loop.run_in_executor(None, ...)`) so the blocking Roboflow call does not
block the event loop.

```
async def _analysis_loop():
    loop = asyncio.get_event_loop()
    while True:
        frame, meta = _get_latest_frame()   # returns (None, None) if no frame yet
        if frame is None:
            await asyncio.sleep(0.2)
            continue
        # run blocking work in thread pool
        result = await loop.run_in_executor(None, _run_analysis, frame, meta)
        if result is not None:
            _store_result(result)
        # no sleep — immediately loop
```

`_run_analysis(frame, meta)`:

1. Call `roboflow_workflow.try_predict_for_gate(frame, width, height)`.
   If returns None → return None.
2. Call `laptop_scene.analyze_with_laptop_gate(latest_frame, width, height,
cached_laptop_detection=(det, "roboflow", reliable))`.
   Use the LATEST frame (re-fetch inside this function) for step 2, not
   the frame from step 1.
3. Return the result dict.

### Endpoints

**POST /api/frame**
Store frame in `_latest_frame`. Return `{"status": "ok"}`.

**GET /api/latest-result**
Return `_latest_result` if set, else `{"status": "pending"}`.

**GET /api/health**

```json
{
  "status": "ok",
  "roboflow_workflow_gate": bool,
  "roboflow_workflow_configured": bool,
  "has_frame": bool,
  "has_result": bool,
  "client_data_poll_interval_sec": 2,
  "client_alert_cooldown_sec": 20
}
```

**POST /api/collect**
Body adds `distance_cm: float` to FrameRequest. Save frame to disk via
`collector.save_frame(frame, distance_cm)`. Return `{"status": "ok", "path": ...}`.

**GET /api/stats**
Return `collector.get_stats()`.

---

## roboflow_workflow.py

Wraps `inference_sdk.InferenceHTTPClient`. Singleton client, created once.

**Functions:**

`gate_enabled() -> bool`
Returns True if env `LAPTOP_USE_ROBOFLOW_WORKFLOW` is "1"/"true"/"yes"/"on".

`configured() -> bool`
Returns True if `ROBOFLOW_API_KEY`, `ROBOFLOW_WORKSPACE`, `ROBOFLOW_WORKFLOW_ID`
are all non-empty.

`try_predict_for_gate(frame: np.ndarray, width: int, height: int) -> dict | None`

- If not `gate_enabled()` or not `configured()` → return None.
- Convert grayscale frame to BGR with `cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)`.
- Call `client.run_workflow(workspace_name=..., workflow_id=..., images={image_key: bgr})`.
- Parse response with `detection_from_workflow_result(raw, width, height)`.
- On any exception → print error, return None.

`detection_from_workflow_result(result, frame_w, frame_h) -> dict`
Parse the Roboflow JSON (arbitrary structure). Recursively walk the response
collecting `(label, confidence)` pairs. Look for keys: `class`, `class_name`,
`label`, `predicted_class`, `top`, `detection_class` for label; `confidence`,
`score`, `probability` for confidence.

Find the best laptop-matching label (check env `ROBOFLOW_LAPTOP_LABELS`,
default "laptop"; split by comma; check if any substring matches label.lower()).

Also try to extract a bounding box: look for dicts with label + width/height +
either center (x,y) or (x_min,y_min,x_max,y_max). Support both pixel and
normalized (0–1) coordinates. Scale to frame pixels. Clamp to frame bounds.

Return:

```python
{
  "class_id": 0,          # 0 if laptop found, -1 if not
  "class_name": str,
  "confidence": float,    # rounded 4 decimals
  "top3": [...],          # top 3 label/confidence pairs
  "source": "roboflow_workflow",
  "bbox": {"x": int, "y": int, "w": int, "h": int}  # only if detection box found
}
```

---

## laptop_scene.py

**Constants / config:**

```python
LAPTOP_CONFIDENCE_MIN = float(os.environ.get("LAPTOP_CONFIDENCE_MIN", "0.4"))
_LAPTOP_NAME_HINTS = frozenset(
    ("laptop", "notebook", "monitor", "screen", "lcd", "computer", "display")
)
```

**`_is_laptop(det: dict) -> bool`**
Returns True if `confidence >= LAPTOP_CONFIDENCE_MIN`.

**`_resolve_laptop_roi(frame, width, height, det) -> tuple | None`**
Returns `(x, y, bw, bh, bbox_area_ratio_pct, cref, bbox_source)` or None.

- If `det` has `"bbox"` and source is roboflow_workflow: use it.
- Else: call `screen_distance.find_largest_bright_bbox(frame, width, height)`.
- `cref`: 0.80 if bbox aspect ratio is 1.2–2.5 (laptop-shaped), else 0.50.
- `bbox_area_ratio_pct = 100 * (bw * bh) / (width * height)`.

**`analyze_with_laptop_gate(frame, width, height, *, cached_laptop_detection) -> dict | None`**

- `cached_laptop_detection` is `(det, detection_source, detection_reliable)`.
  If provided, skip calling Roboflow.
- If not `_is_laptop(det)` → return `{"laptop_detected": False, ...}`.
- Call `_resolve_laptop_roi(...)`. If None → return `{"laptop_detected": True, "analysis_ready": False, "reason": "no_screen_blob", ...}`.
- Compute distance: `screen_distance.compute_distance(bbox_area_ratio_pct, cref)`.
- Apply EMA: `temporal_smoothing.apply_smoothing(raw_d, raw_r)`.
- Compute brightness: `brightness.analyze_brightness_bbox(frame, x, y, bw, bh, width, height)`.
- Return merged dict:

```python
{
  "laptop_detected": True,
  "analysis_ready": True,
  "laptop_class": str,
  "laptop_confidence": float,
  "bbox": {"x", "y", "w", "h"},
  "bbox_source": "roboflow_workflow" | "bright_blob",
  "distance_cm": float,
  "category": "TOO_CLOSE" | "OK" | "FAR",
  "screen_ratio": int,
  "screenBr": float,
  "ambientBr": float,
  "contrast": float,
  "brightnessAlert": "DULL" | "MODERATE" | "BRIGHT",
  "detection_source": str,
  "detection_reliable": bool,
}
```

---

## screen_distance.py

Read calibration from env (with float defaults):

```
DISTANCE_D_REF = 60.0       # cm at reference fill ratio
DISTANCE_REF_RATIO_PCT = 25.0
DISTANCE_MIN_RATIO_PCT = 0.5
DISTANCE_CM_MIN = 20.0
DISTANCE_CM_MAX = 150.0
```

**`find_largest_bright_bbox(frame, width, height) -> tuple | None`**

- GaussianBlur(5,5) → Otsu threshold.
- If foreground mean < background mean, invert (keeps bright region as fg).
- Find contours, take largest with area ≥ max(50, 1% of frame).
- Return `(x, y, bw, bh, bbox_area_ratio_pct, cref)`.

**`compute_distance(bbox_area_ratio_pct, cref) -> dict`**
Inverse sqrt model:

```python
distance_cm = D_REF * sqrt(REF_RATIO / max(ratio, MIN_RATIO))
distance_cm = clip(distance_cm, CM_MIN, CM_MAX)
```

Return `{"distance_cm": float, "category": str, "confidence": cref, "source": "laptop_bbox", "screen_ratio": int, "bbox_area_ratio_pct": float}`.

**`category_from_distance_cm(cm) -> str`**
`< 35` → "TOO_CLOSE", `> 65` → "FAR", else "OK".

---

## brightness.py

**`analyze_brightness_bbox(frame, x, y, bw, bh, width, height) -> dict`**

- Screen brightness = mean of pixels inside bbox.
- Ambient brightness = mean of pixels outside bbox.
  If outside pixels < 5% of frame area, fall back to darkest of 4 corner
  patches (each `h//6 × w//6`).
- Scale both to 0–100 percent.
- `contrast = abs(screenBr - ambientBr)`.
- `brightnessAlert`: "DULL" if contrast < 10, "BRIGHT" if > 25, else "MODERATE".
- Return `{"screenBr", "ambientBr", "contrast", "brightnessAlert"}`.

---

## temporal_smoothing.py

EMA with alpha=0.4 on distance_cm and bbox_area_ratio_pct.
Global state (single device). First frame seeds the filter (no blending).

```python
EMA_ALPHA = 0.4

def apply_smoothing(distance_cm, bbox_area_ratio_pct) -> tuple[float, float]:
    ...  # returns (smoothed_d, smoothed_r), both rounded
```

---

## collector.py

**`save_frame(frame: np.ndarray, distance_cm: float) -> dict`**

- Bucket `distance_cm` into class labels: `< 35` → "too_close", `35–65` → "ok", `> 65` → "far".
- Save raw bytes to `dataset/<class>/frame_<timestamp>.raw`.
- Return `{"status": "saved", "class": str, "path": str}`.

**`get_stats() -> dict`**
Return counts per class from `dataset/` directory.

---

## CORS

Allow all origins, GET and POST methods.

---

## Run

```
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
