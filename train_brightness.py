"""
Train a brightness regressor (TFLite) from collected dataset images.

Uses the same folders as train.py: dataset/{N}cm/*.jpg (160×120 grayscale JPEGs).
Targets are screenBr / ambientBr from brightness.analyze_brightness on each image
(bootstrap). If there are no JPEGs yet, the script **generates synthetic 120×160
frames** so training still produces a TFLite file (smoke / dev only — retrain with
real `/api/collect` data for production).

Output: models/brightness_model.tflite

Usage
-----
    pip install "tensorflow>=2.16"
    python train_brightness.py
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np

import brightness as bright

DATASET_DIR = Path(__file__).parent / "dataset"
MODEL_OUT = Path(__file__).parent / "models" / "brightness_model.tflite"

IMG_W, IMG_H = 96, 96
BATCH_SIZE = 32
EPOCHS = 35
VAL_SPLIT = 0.2
SEED = 42
# When dataset/*cm/*.jpg is empty, generate this many synthetic frames (dev bootstrap).
SYNTHETIC_N = 256


def _synthetic_bootstrap(n: int) -> tuple[list[np.ndarray], list[list[float]]]:
    """Random bright-centre / darker-corner patterns at collector resolution (160×W, 120×H)."""
    rng = np.random.default_rng(SEED)
    imgs: list[np.ndarray] = []
    y_br: list[list[float]] = []
    h, w = 120, 160
    for _ in range(n):
        img = rng.integers(10, 45, size=(h, w), dtype=np.uint8)
        r0, r1 = h // 4, h - h // 4
        c0, c1 = w // 4, w - w // 4
        center_lv = int(rng.integers(70, 245))
        img[r0:r1, c0:c1] = center_lv
        ch, cw = max(3, h // 6), max(3, w // 6)
        cmax = max(5, min(center_lv - 10, 130))
        for y0, x0 in ((0, 0), (0, w - cw), (h - ch, 0), (h - ch, w - cw)):
            img[y0 : y0 + ch, x0 : x0 + cw] = int(rng.integers(5, cmax))

        b = bright.analyze_brightness(img, w, h)
        small = cv2.resize(img, (IMG_W, IMG_H)).astype(np.float32) / 255.0
        imgs.append(small)
        y_br.append([b["screenBr"] / 100.0, b["ambientBr"] / 100.0])
    return imgs, y_br


def load_dataset():
    images, y_br = [], []

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    for label_dir in sorted(DATASET_DIR.iterdir()):
        if not label_dir.is_dir() or not label_dir.name.endswith("cm"):
            continue
        for img_path in label_dir.glob("*.jpg"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            h, w = img.shape[:2]
            b = bright.analyze_brightness(img, w, h)
            img = cv2.resize(img, (IMG_W, IMG_H)).astype(np.float32) / 255.0
            images.append(img)
            y_br.append([b["screenBr"] / 100.0, b["ambientBr"] / 100.0])

    if not images:
        print(
            "[train_brightness] No JPEGs under dataset/*cm/ — using synthetic bootstrap "
            f"({SYNTHETIC_N} images). Add real frames via POST /api/collect and retrain."
        )
        images, y_br = _synthetic_bootstrap(SYNTHETIC_N)

    X = np.array(images)[..., np.newaxis]
    y = np.array(y_br, dtype=np.float32)

    rng = random.Random(SEED)
    idx = list(range(len(X)))
    rng.shuffle(idx)
    X, y = X[idx], y[idx]

    split = int(len(X) * (1 - VAL_SPLIT))
    return X[:split], y[:split], X[split:], y[split:]


def build_model():
    import tensorflow as tf  # noqa: PLC0415
    from tensorflow.keras import layers, models  # noqa: PLC0415

    inp = layers.Input(shape=(IMG_H, IMG_W, 1))
    x = layers.Lambda(lambda t: tf.repeat(t, 3, axis=-1))(inp)
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_H, IMG_W, 3),
        include_top=False,
        weights="imagenet",
        alpha=0.35,
    )
    base.trainable = False
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(2, activation="sigmoid")(x)
    model = models.Model(inp, out)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model, base


def export_tflite(model) -> None:
    import tensorflow as tf  # noqa: PLC0415

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    MODEL_OUT.write_bytes(converter.convert())
    print(f"[train_brightness] TFLite → {MODEL_OUT} ({MODEL_OUT.stat().st_size // 1024} KB)")


def main():
    import tensorflow as tf  # noqa: PLC0415

    print("[train_brightness] Loading dataset + pseudo-labels from brightness.py…")
    X_train, y_train, X_val, y_val = load_dataset()
    print(f"         train={len(X_train)} val={len(X_val)}")

    model, base = build_model()
    cb = [
        tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1),
    ]
    print("[train_brightness] Phase 1 — frozen backbone…")
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=cb,
        verbose=1,
    )

    print("[train_brightness] Phase 2 — fine-tune top MobileNet layers…")
    base.trainable = True
    for layer in base.layers[:-20]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="mse", metrics=["mae"])
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=12,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)],
        verbose=1,
    )

    export_tflite(model)
    print("[train_brightness] Done. POST /api/reload to load (or restart server).")


if __name__ == "__main__":
    main()
