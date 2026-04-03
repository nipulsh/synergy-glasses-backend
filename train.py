"""
Train a distance classifier from collected frames, then export to TFLite.

Usage
-----
    cd backend
    pip install tensorflow>=2.16   # only needed for training
    python train.py

Input:  dataset/{N}cm/*.jpg    (collected via the phone app's Collect screen)
Output: models/model.tflite    (loaded automatically by inference.py)

The model
---------
MobileNetV2 with a grayscale input (H×W×1) → 3-class softmax output:
    0 = TOO_CLOSE  (< 35 cm)
    1 = OK         (35–65 cm)
    2 = FAR        (> 65 cm)

You can also train a regression model by setting REGRESSION=True below.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Config — adjust to taste
# ---------------------------------------------------------------------------
DATASET_DIR = Path(__file__).parent / "dataset"
MODEL_OUT   = Path(__file__).parent / "models" / "model.tflite"

IMG_W, IMG_H = 96, 96        # input size for MobileNetV2
BATCH_SIZE   = 32
EPOCHS       = 30
VAL_SPLIT    = 0.2
REGRESSION   = False         # True = predict cm directly; False = 3-class

# Distance → class label mapping (for classification)
def _to_class(cm: int) -> int:
    if cm < 35:
        return 0   # TOO_CLOSE
    if cm <= 65:
        return 1   # OK
    return 2       # FAR


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset():
    images, labels = [], []

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")

    for label_dir in sorted(DATASET_DIR.iterdir()):
        if not label_dir.is_dir():
            continue
        name = label_dir.name  # e.g. "40cm"
        if not name.endswith("cm"):
            continue
        try:
            cm = int(name[:-2])
        except ValueError:
            continue

        for img_path in label_dir.glob("*.jpg"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_W, IMG_H)).astype(np.float32) / 255.0
            images.append(img)
            labels.append(cm if REGRESSION else _to_class(cm))

    if not images:
        raise ValueError("No images found in dataset. Collect frames with the app first.")

    X = np.array(images)[..., np.newaxis]   # (N, H, W, 1)
    y = np.array(labels, dtype=np.float32)

    # Shuffle
    idx = list(range(len(X)))
    random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    split = int(len(X) * (1 - VAL_SPLIT))
    return X[:split], y[:split], X[split:], y[split:]


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

def build_model():
    import tensorflow as tf  # noqa: PLC0415
    from tensorflow.keras import layers, models  # noqa: PLC0415

    inp = layers.Input(shape=(IMG_H, IMG_W, 1))

    # Convert 1-channel to 3-channel so MobileNetV2 accepts it
    x = layers.Lambda(lambda t: tf.repeat(t, 3, axis=-1))(inp)

    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_H, IMG_W, 3),
        include_top=False,
        weights="imagenet",
        alpha=0.35,   # smallest variant
    )
    base.trainable = False   # feature extraction only (fast training)

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)

    if REGRESSION:
        out = layers.Dense(1, activation="relu")(x)
        model = models.Model(inp, out)
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    else:
        out = layers.Dense(3, activation="softmax")(x)
        model = models.Model(inp, out)
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    return model


# ---------------------------------------------------------------------------
# Export to TFLite
# ---------------------------------------------------------------------------

def export_tflite(model) -> Path:
    import tensorflow as tf  # noqa: PLC0415

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # quantise weights
    tflite_model = converter.convert()

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.write_bytes(tflite_model)
    print(f"\n[train] TFLite model saved → {MODEL_OUT}  ({len(tflite_model)//1024} KB)")
    return MODEL_OUT


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("[train] Loading dataset…")
    X_train, y_train, X_val, y_val = load_dataset()
    print(f"[train] Train: {len(X_train)}  Val: {len(X_val)}")

    classes, counts = np.unique(y_train.astype(int) if not REGRESSION else np.zeros(1), return_counts=True)
    if not REGRESSION:
        label_names = {0: "TOO_CLOSE", 1: "OK", 2: "FAR"}
        for c, n in zip(classes, counts):
            print(f"         class {label_names.get(int(c), c)}: {n} frames")

    print("[train] Building model…")
    model = build_model()

    import tensorflow as tf  # noqa: PLC0415

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1),
    ]

    print("[train] Training…")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    export_tflite(model)
    print("[train] Done. Restart the backend server to load the new model.")
    print("         Or call POST /api/reload to hot-reload without restart.")


if __name__ == "__main__":
    main()
