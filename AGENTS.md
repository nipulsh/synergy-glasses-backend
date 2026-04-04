# Synergy / OcuSmart glasses backend — agent context

## Purpose

FastAPI service that receives **raw grayscale camera frames** from smart glasses (ESP32-S3 + OV2640, typically **80×60** uint8) and returns **screen distance** estimates, optional **brightness/contrast** analysis, and endpoints to **collect training data** and **train** a TFLite model.

Branding in code: **OcuSmart Distance API** (`main.py`). Repository name: **synergy-glasses-backend**.

## Stack

- **Python 3.11** in Docker (`Dockerfile`); local dev can use any version that satisfies `requirements.txt`.
- **FastAPI** + **Uvicorn** (`main.py`).
- **NumPy**, **OpenCV (headless)** for image ops and heuristics.
- **TFLite** for ML inference: prefer `tflite-runtime` when available; else **TensorFlow**’s `tf.lite.Interpreter` (`inference.py`). If no model file exists, inference falls back to **OpenCV + heuristics** (no TF needed at runtime for that path).
- **Training** (`train.py`) requires **TensorFlow/Keras** (not in `requirements.txt` — install separately).

## Layout (repo root = app root)

| File | Role |
|------|------|
| `main.py` | FastAPI app, routes, frame decode (base64 → `uint8` H×W). |
| `inference.py` | Load `models/model.tflite`; `predict_distance()`; `reload_model()`. |
| `object_inference.py` | Load `models/objects_model.tflite` + `objects_class_names.json`; `predict_objects()`; `reload_objects_model()`. |
| `laptop_scene.py` | Gate `/api/distance` and `/api/analyze` when **Roboflow workflow** (`LAPTOP_USE_ROBOFLOW_WORKFLOW`) and/or **`objects_model.tflite`** is active; strict pipeline: TFLite objects first when model present, bbox inverse-sqrt distance, bbox-split brightness, EMA smoothing. |
| `temporal_smoothing.py` | Process-global **EMA** (default **α = 0.4**) on **`distance_cm`** and **`bbox_area_ratio_pct`** after raw bbox computation on the laptop-gated success path. |
| `roboflow_workflow.py` | Optional **`InferenceHTTPClient`** singleton + **in-memory** image → workflow (`ROBOFLOW_*` env). Used only as **fallback** when the objects TFLite model is **not** loaded. Training uses `train_objects.py`, not this API. |
| `screen_distance.py` | **edge_heuristic** (contour area%) when no distance TFLite; laptop-gated bbox path uses **inverse square-root** fill-ratio model (`DISTANCE_D_REF`, `DISTANCE_REF_RATIO_PCT`, clamps) via **`compute_distance_from_fill_ratio_pct`**. |
| `brightness.py` | Centre vs corners heuristic; **`analyze_brightness_bbox`** (inside rect vs outside / corner fallback); **`pack_brightness`**. |
| `brightness_inference.py` | Prefer **`models/brightness_model.tflite`** (`train_brightness.py`); else delegates to `brightness.py`. |
| `collector.py` | Save labeled frames under `dataset/{N}cm/*.jpg`; `get_stats()`. |
| `train.py` | MobileNetV2 distance classifier (or `REGRESSION=True` for cm) → `models/model.tflite`. |
| `train_brightness.py` | MobileNetV2 regressor (2× sigmoid) on `dataset/*cm/*.jpg` → `models/brightness_model.tflite`. |
| `train_objects.py` | Kaggle, pickle, or **`--source folder --data-dir`** (e.g. Roboflow Classification export `train/`) → `models/objects_model.tflite` (+ `.keras`, class names JSON). |
| `Dockerfile` / `docker-entrypoint.sh` | Production image; mount `/data` to persist `dataset/` and `models/` (symlinks created at container start). |

## HTTP API (`/api/*`)

All POST bodies that carry a frame use JSON:

```json
{
  "frame": "<base64 of raw grayscale bytes, length width×height>",
  "width": 80,
  "height": 60
}
```

- `GET /api/health` — distance / objects / brightness model flags and paths (`*_model_loaded`, `*_model_path`); **`roboflow_workflow_gate`**, **`roboflow_workflow_configured`**.
- `POST /api/distance` — If **Roboflow workflow gate** is on (see `.env.example`) **or** **`objects_model.tflite` is loaded**: laptop gate (`laptop_scene`). Not a laptop → **`{"laptop_detected": false}`** only (plus **`detection_source`** / **`detection_reliable`**). Laptop + bright screen blob → **`laptop_bbox`** distance + **`brightness_source: laptop_bbox`**. If **neither** Roboflow gate nor objects model: legacy distance + brightness (`tflite` \| `heuristic`).
- `POST /api/analyze` — Same gating as `/api/distance`; `include_objects=1` adds **`laptop_top3`** (and **`laptop_source`** when Roboflow). Legacy path still adds **`objects`** when only TFLite is used. Responses include **`detection_source`** (`tflite` \| `roboflow` \| `heuristic` \| `none`) and **`detection_reliable`** where applicable.
- `POST /api/objects` — object/laptop classifier (`object_inference.predict_objects`); **503** if objects model missing; includes **`detection_source`** / **`detection_reliable`**.
- `POST /api/collect` — same as frame request plus `distance_cm` (5–200); saves to dataset (`collector.save_frame`).
- `GET /api/stats` — dataset totals per class folder.
- `POST /api/reload` — hot-reload **distance** and **brightness** TFLite models.
- `POST /api/reload-objects` — hot-reload `objects_model.tflite` after `train_objects.py`.

CORS is currently **allow all** origins in `main.py`. `.env.example` documents `PORT` / `MODEL_PATH` / `CORS_ORIGINS` / laptop gate / **`ROBOFLOW_*`**; the running app does **not** load `.env` by default — use env vars your host sets or extend the app if needed.

## Inference behavior

**Laptop gate** when **`LAPTOP_USE_ROBOFLOW_WORKFLOW=1`** plus `ROBOFLOW_API_KEY` / `ROBOFLOW_WORKSPACE` / `ROBOFLOW_WORKFLOW_ID`, and/or **`objects_model.tflite` is present**. **When the objects TFLite model is loaded**, **`object_inference`** is the **only** per-request laptop signal — **Roboflow is not called** (singleton client + in-memory image remain for **fallback** when the model file is absent but the workflow gate and credentials are set). Roboflow path needs **`inference-sdk`** (Python **&lt;3.13**). Workflow JSON is scanned for label/confidence pairs; labels matching **`ROBOFLOW_LAPTOP_LABELS`** (default `laptop`) set the score checked against **`LAPTOP_CONFIDENCE_MIN`** (default **0.4**, typically **0.35–0.45**). For TFLite, **`LAPTOP_CLASS_IDS`** / name hints apply as before. **`detection_reliable`**: heuristic → false; TFLite → true when confidence ≥ threshold; Roboflow → true when used. On the gated **success** path, distance/brightness TFLite heads are **not** used (bbox inverse-sqrt distance + bbox-split brightness + EMA); legacy path still uses distance/brightness models when no gate fires.

**Training with a Roboflow Classification export:** unzip so you have `…/train/<ClassName>/*.jpg`, then `python train_objects.py --source folder --data-dir path/to/train` → `models/objects_model.tflite` for local inference without per-frame Roboflow calls.

**Legacy (no Roboflow gate and no objects model):**

1. If `models/model.tflite` loads: **source `tflite`**. Supports **regression** (single float → `distance_cm` clipped 10–120) or **3-class softmax** → mapped categories TOO_CLOSE / OK / FAR with representative cm values.
2. Else: **source `edge_heuristic`** via `screen_distance.analyze_screen_distance` (includes `screen_ratio`).
3. Brightness: if `models/brightness_model.tflite` loads, **source `tflite`** (regressed `screenBr` / `ambientBr`); else heuristic ROIs in `brightness.py`.

Distance categories align with training: TOO_CLOSE &lt; 35 cm, OK 35–65 cm, FAR &gt; 65 cm (`inference.py`, `train.py`).

## Data & training

- Collection rounds labels to **nearest 5 cm** bin; images saved as JPEG upscaled to **160×120** (`collector.py`).
- `train.py` reads `dataset/*cm/*.jpg`, resizes to **96×96** grayscale, exports quantized TFLite to `models/model.tflite`.
- `train_objects.py --source folder --data-dir …` trains the laptop/object classifier from folder-per-class exports (including Roboflow Classification).
- After deploy: call `POST /api/reload` or restart to pick up a new model.

## Deployment (no Fly.io config in repo)

- **Docker:** `docker build` / `docker run` from repo root; many hosts inject `PORT` (see `Dockerfile` `CMD`). For persistent collected data and models, mount a volume at **`/data`** (matches `docker-entrypoint.sh`).
- **Common platforms:** [Railway](https://railway.app), [Render](https://render.com), [Google Cloud Run](https://cloud.google.com/run), [Azure Container Apps](https://azure.microsoft.com/products/container-apps), [DigitalOcean App Platform](https://www.digitalocean.com/products/app-platform), or any VPS with Docker or `uvicorn` behind Caddy/nginx.

## Run locally

```bash
pip install -r requirements.txt
# optional: pip install tflite-runtime  # or tensorflow for inference
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

- On **Windows** (PowerShell), if `uvicorn` is not recognized, use **`python -m uvicorn`** with the same arguments (the venv `Scripts` folder may not be on `PATH`).
- **Tunnels** (ngrok, etc.): forward to **8000** to match Uvicorn’s default; port **80** only works if something is listening there.
- **Editor-integrated terminals**: if a newly installed CLI is still missing after a new tab, **fully restart the editor** so terminals inherit an updated user `Path`, or reload `Path` from User/Machine env for that session.

Training (TensorFlow required):

```bash
pip install "tensorflow>=2.16"
python train.py
# Objects / laptop classifier — in Explorer open the `train` folder, copy the address bar, paste inside quotes:
python train_objects.py --source folder --data-dir "C:\Users\YourName\Downloads\YourRoboflowExport\train"
```

## Conventions for changes

- Keep **frame shape** contract consistent with clients: default **80×60** row-major grayscale; validate byte length = `width * height` in `main.py`.
- New endpoints: follow existing **Pydantic** models and **HTTPException** patterns in `main.py`.
- Prefer **small, focused edits**; do not rename public JSON keys (`distance_cm`, `screenBr`, etc.) without coordinating mobile/firmware clients.

## Learned User Preferences

- On Windows, if `uvicorn` is not on PATH, use `python -m uvicorn main:app --host 0.0.0.0 --port 8000` (and `--reload` for dev).
- Point tunnels (ngrok, etc.) at the **same port as Uvicorn** (default **8000**), not port 80, unless something is actually listening on 80.
- After installing CLIs or changing user PATH, **fully restart Cursor** or refresh `Path` in the terminal session if integrated terminals still cannot resolve new commands.
- TensorFlow training (`train.py`, `train_objects.py`) is most reliable on **Python 3.10–3.12**; if 3.13 wheels fail, use a separate venv with a supported Python for training only.

## Learned Workspace Facts

- **Edge heuristic:** `screen_distance.py` treats a **larger** Otsu bright-blob `screen_ratio` as **closer** (lower `distance_cm`). A prior `100 - ratio` variant incorrectly made distance rise with blob size.
- **Ambient brightness:** `brightness.py` uses the minimum mean of four corner patches, not a thin border ring (the ring often matched the LCD and produced near-zero contrast).
- **Docker on Linux (e.g. Render):** CRLF in `docker-entrypoint.sh` can cause `exec … no such file or directory`; the Dockerfile strips `\r` after copy, `.gitattributes` keeps `*.sh` as LF, and `ENTRYPOINT` runs the script via `/bin/sh`.
- **Config:** the app does not load `.env` by default; `PORT` in the Docker CMD is what most hosts inject. `.env.example` documents optional vars that are not wired until code reads them.
- **Two TFLite heads:** Distance → `models/model.tflite` (`inference.py`, `POST /api/reload`). Objects/laptop classes → `models/objects_model.tflite` + `objects_class_names.json` (`object_inference.py`, `POST /api/reload-objects`). Same grayscale frame can feed both; each interpreter supplies its own input size.
- **Grayscale → RGB for MobileNet-style inputs:** The glasses payload stays single-channel; code **replicates gray to three channels** (`COLOR_GRAY2RGB` or equivalent) for 3-channel TFLite models — not true color capture.
- **`train_objects.py`:** Default source is **kagglehub** (cached laptop image set under `data/images/c*/`); use **`requirements-train.txt`** for TF + kagglehub; ensure **`models/`** exists before `model.save` / TFLite export (Keras errors if the directory is missing).
- **Object model domain:** Classifier was trained on **catalog-style** laptop photos; expect **weak transfer** to **80×60 wearable POV** until fine-tuned on glasses-like data.
- **`temporal_smoothing`:** EMA state is **process-global**; multiple devices or clients hitting one server instance **share** the same smoother unless you key by client/session or call **`reset_smoothing_state`** where appropriate.
