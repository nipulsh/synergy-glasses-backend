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
| `screen_distance.py` | Fallback **edge_heuristic**: Otsu blob area → `screen_ratio`; distance/category use `close_signal = 100 - ratio` (hardware inverts naive blob-size vs distance). |
| `brightness.py` | Screen-centre vs **darkest corner patches** (ambient) → `screenBr`, `ambientBr`, `contrast`, `brightnessAlert`. |
| `collector.py` | Save labeled frames under `dataset/{N}cm/*.jpg`; `get_stats()`. |
| `train.py` | MobileNetV2-based classifier (or regression if `REGRESSION=True`) → `models/model.tflite`. |
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

- `GET /api/health` — `status`, `model_loaded`, `model_path`.
- `POST /api/distance` — distance prediction (`inference.predict_distance`).
- `POST /api/analyze` — distance + brightness (`brightness.analyze_brightness`).
- `POST /api/collect` — same as frame request plus `distance_cm` (5–200); saves to dataset (`collector.save_frame`).
- `GET /api/stats` — dataset totals per class folder.
- `POST /api/reload` — hot-reload TFLite after training.

CORS is currently **allow all** origins in `main.py`. `.env.example` documents `PORT` / `MODEL_PATH` / `CORS_ORIGINS`; the running app does **not** load `.env` by default — use env vars your host sets or extend the app if needed.

## Inference behavior

1. If `models/model.tflite` loads: **source `tflite`**. Supports **regression** (single float → `distance_cm` clipped 10–120) or **3-class softmax** → mapped categories TOO_CLOSE / OK / FAR with representative cm values.
2. Else: **source `edge_heuristic`** via `screen_distance.analyze_screen_distance` (includes `screen_ratio`).

Distance categories align with training: TOO_CLOSE &lt; 35 cm, OK 35–65 cm, FAR &gt; 65 cm (`inference.py`, `train.py`).

## Data & training

- Collection rounds labels to **nearest 5 cm** bin; images saved as JPEG upscaled to **160×120** (`collector.py`).
- `train.py` reads `dataset/*cm/*.jpg`, resizes to **96×96** grayscale, exports quantized TFLite to `models/model.tflite`.
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
```

## Conventions for changes

- Keep **frame shape** contract consistent with clients: default **80×60** row-major grayscale; validate byte length = `width * height` in `main.py`.
- New endpoints: follow existing **Pydantic** models and **HTTPException** patterns in `main.py`.
- Prefer **small, focused edits**; do not rename public JSON keys (`distance_cm`, `screenBr`, etc.) without coordinating mobile/firmware clients.

## Learned User Preferences

- On Windows, if `uvicorn` is not on PATH, use `python -m uvicorn main:app --host 0.0.0.0 --port 8000` (and `--reload` for dev).
- Point tunnels (ngrok, etc.) at the **same port as Uvicorn** (default **8000**), not port 80, unless something is actually listening on 80.
- After installing CLIs or changing user PATH, **fully restart Cursor** or refresh `Path` in the terminal session if integrated terminals still cannot resolve new commands.

## Learned Workspace Facts

- **Edge heuristic:** blob-area vs physical distance is inverted for this glasses/FOV; `screen_distance.py` maps categories and `distance_cm` from `close_signal = 100 - screen_ratio` while still returning raw `screen_ratio`. TFLite behavior is unchanged.
- **Ambient brightness:** `brightness.py` uses the minimum mean of four corner patches, not a thin border ring (the ring often matched the LCD and produced near-zero contrast).
- **Docker on Linux (e.g. Render):** CRLF in `docker-entrypoint.sh` can cause `exec … no such file or directory`; the Dockerfile strips `\r` after copy, `.gitattributes` keeps `*.sh` as LF, and `ENTRYPOINT` runs the script via `/bin/sh`.
- **Config:** the app does not load `.env` by default; `PORT` in the Docker CMD is what most hosts inject. `.env.example` documents optional vars that are not wired until code reads them.
