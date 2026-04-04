"""
Train a laptop / object classifier and export to TFLite.

Data sources
------------
1) **Kaggle (default)** — downloads `sugxm00/laptop-dataset` via kagglehub
   (needs `~/.kaggle/kaggle.json` or `KAGGLE_API_TOKEN`).
   Layout: `<cache>/.../data/images/c0/*.jpg`, `c1/`, … class = folder name.

2) **Pickle** — `X.pickle/X.pickle` + `X.pickle/y.pickle` (see `--source pickle`).

3) **Folder** — local directory whose **immediate subfolders** are class names (e.g. Roboflow
   **Classification** → Generate → download ZIP → unzip → point `--data-dir` at `train/` or the
   folder that contains `Laptop/`, `Chair/`, …). Same layout as Kaggle `images/<class>/*`.

Usage
-----
    pip install "tensorflow>=2.16" kagglehub
    python train_objects.py              # Kaggle laptop dataset
    python train_objects.py --source pickle
    python train_objects.py --source folder --data-dir "D:\\datasets\\roboflow\\train"

Output: models/objects_model.tflite, models/objects_model.keras,
        models/objects_class_names.json

Optional: set TRAIN_OBJECTS_EPOCHS=2 (or similar) for a quick local smoke test only.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
KAGGLE_DATASET = "sugxm00/laptop-dataset"

DATA_DIR = Path(__file__).parent / "X.pickle"
X_PATH = DATA_DIR / "X.pickle"
Y_PATH = DATA_DIR / "y.pickle"

MODEL_OUT = Path(__file__).parent / "models" / "objects_model.tflite"
CLASS_NAMES_OUT = Path(__file__).parent / "models" / "objects_class_names.json"

IMG_SIZE = 160
BATCH_SIZE = 16
# Override for quick smoke tests, e.g. TRAIN_OBJECTS_EPOCHS=2
EPOCHS = max(1, int(os.environ.get("TRAIN_OBJECTS_EPOCHS", "40")))
VAL_SPLIT = 0.2
SEED = 42
ALPHA = 0.35

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _dir_has_image(d: Path) -> bool:
    try:
        for f in d.iterdir():
            if f.is_file() and f.suffix.lower() in IMAGE_EXT:
                return True
    except OSError:
        pass
    return False


def _looks_like_doc_placeholder_path(path_str: str) -> bool:
    """True if the user pasted an example path from the docs instead of a real folder."""
    t = path_str.strip().lower().replace("/", "\\")
    markers = (
        "where_you_unzipped",
        "your_export",
        "path\\to\\train",
        "replace_with",
        "<your",
        "paste-here",
        "paste_full_path_here",
        "example\\datasets",
        "the_path_you_copied",
        "your_actual_path",
        "full_path_here",
    )
    if any(m in t for m in markers):
        return True
    # e.g. ...\synergy-glasses-backend\THE_PATH_YOU_COPIED
    return Path(path_str.replace("\\", "/")).name.lower() == "the_path_you_copied"


def _download_kaggle_dataset() -> Path:
    import kagglehub  # noqa: PLC0415

    path = kagglehub.dataset_download(KAGGLE_DATASET)
    root = Path(path)
    print(f"[train_objects] Kaggle dataset path: {root}")
    return root


def _sorted_class_names(images_dir: Path) -> list[str]:
    """Folder names like c0, c1, … c16 sorted numerically, not lexicographically."""
    dirs = [p.name for p in images_dir.iterdir() if p.is_dir()]

    def sort_key(name: str) -> tuple:
        if len(name) > 1 and name[0] == "c" and name[1:].isdigit():
            return (0, int(name[1:]))
        return (1, name)

    return sorted(dirs, key=sort_key)


def _sorted_class_names_with_images(images_dir: Path) -> list[str]:
    """
    Subfolders that actually contain ≥1 image file, skipping hidden dirs (e.g. `.git`).
    Must match what Keras image_dataset_from_directory indexes — do not pass unrelated folders.
    """
    names: list[str] = []
    for p in images_dir.iterdir():
        if not p.is_dir() or p.name.startswith("."):
            continue
        if _dir_has_image(p):
            names.append(p.name)

    def sort_key(name: str) -> tuple:
        if len(name) > 1 and name[0] == "c" and name[1:].isdigit():
            return (0, int(name[1:]))
        return (1, name)

    return sorted(names, key=sort_key)


def _find_image_classification_root(dataset_path: Path) -> Path:
    """Find a directory whose immediate subfolders each hold images (class folders)."""
    candidates = [
        dataset_path / "data" / "images",
        dataset_path / "images",
        dataset_path,
    ]
    for c in candidates:
        if not c.is_dir():
            continue
        subdirs = [p for p in c.iterdir() if p.is_dir()]
        if len(subdirs) < 2:
            continue

        if sum(1 for d in subdirs if _dir_has_image(d)) >= 2:
            return c
    raise FileNotFoundError(
        f"Could not find class folders with images under {dataset_path}. "
        "Expected something like data/images/<class>/*.(jpg|png|…)."
    )


def _load_xy_pickle():
    if not X_PATH.exists():
        raise FileNotFoundError(f"Missing {X_PATH}.")
    if not Y_PATH.exists():
        raise FileNotFoundError(
            f"Missing {Y_PATH}. Add y.pickle with one label per row of X, or use "
            "`python train_objects.py` (default) for the Kaggle laptop dataset."
        )

    with open(X_PATH, "rb") as f:
        X = pickle.load(f)
    with open(Y_PATH, "rb") as f:
        y_raw = pickle.load(f)

    class_names: list[str] | None = None
    if isinstance(y_raw, dict):
        class_names = y_raw.get("class_names")
        y_raw = y_raw["y"]

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y_raw).astype(np.int64).reshape(-1)

    if X.ndim != 4 or X.shape[-1] != 3:
        raise ValueError(f"X must be (N, H, W, 3); got {X.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X has {X.shape[0]} samples, y has {y.shape[0]}")

    if X.max() > 1.01 or X.min() < -0.01:
        X = np.clip(X, 0.0, 1.0)

    order = np.sort(np.unique(y))
    num_classes = len(order)
    id_to_idx = {int(v): i for i, v in enumerate(order)}
    y = np.array([id_to_idx[int(v)] for v in y], dtype=np.int64)

    if class_names is not None:
        if len(class_names) != num_classes:
            raise ValueError(
                f"class_names length {len(class_names)} != num_classes {num_classes}"
            )
    else:
        class_names = [str(int(u)) for u in order]

    return X, y, class_names, num_classes


def _split(X, y):
    rng = random.Random(SEED)
    n = len(X)
    idx = list(range(n))
    rng.shuffle(idx)
    split = int(n * (1 - VAL_SPLIT))
    tr, va = idx[:split], idx[split:]
    return X[tr], y[tr], X[va], y[va]


def build_model(num_classes: int):
    import tensorflow as tf  # noqa: PLC0415
    from tensorflow.keras import layers, models  # noqa: PLC0415

    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
        alpha=ALPHA,
    )
    base.trainable = False

    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base


def export_tflite(model) -> None:
    import tensorflow as tf  # noqa: PLC0415

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    MODEL_OUT.write_bytes(converter.convert())
    print(f"[train_objects] TFLite -> {MODEL_OUT} ({MODEL_OUT.stat().st_size // 1024} KB)")


def _train_from_images_dir(images_dir: Path) -> None:
    """Train from a directory of class subfolders (Kaggle layout, Roboflow export, etc.)."""
    import tensorflow as tf  # noqa: PLC0415
    from tensorflow.keras import layers  # noqa: PLC0415

    images_dir = images_dir.resolve()
    if not images_dir.exists():
        hint = ""
        if _looks_like_doc_placeholder_path(str(images_dir)):
            hint = (
                "\n  This path looks like a documentation placeholder. Do not copy example paths - "
                "use File Explorer: open your unzipped Roboflow export, go into the `train` folder, "
                "click the address bar, copy the full path, and pass that to --data-dir.\n"
            )
        raise FileNotFoundError(
            f"Dataset path does not exist: {images_dir}{hint}"
            "  The directory must exist and contain at least two class subfolders with images "
            "(e.g. train/Laptop/*.jpg, train/Chair/*.jpg)."
        )
    if not images_dir.is_dir():
        raise NotADirectoryError(f"Not a directory (maybe a file?): {images_dir}")

    class_names_order = _sorted_class_names_with_images(images_dir)
    if len(class_names_order) < 2:
        raise ValueError(
            f"{images_dir} needs at least two subfolders that each contain image files "
            f"({', '.join(sorted(IMAGE_EXT))}). Hidden folders (names starting with '.') are ignored. "
            "For Roboflow, point --data-dir at the `train` folder (e.g. Laptop/, Chair/), not the repo root."
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        images_dir,
        class_names=class_names_order,
        validation_split=VAL_SPLIT,
        subset="training",
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="int",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        images_dir,
        class_names=class_names_order,
        validation_split=VAL_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="int",
    )

    class_names = list(train_ds.class_names)
    num_classes = len(class_names)
    print(f"[train_objects] Classes ({num_classes}): {class_names[:12]}{'...' if len(class_names) > 12 else ''}")

    norm = layers.Rescaling(1.0 / 255.0)

    def norm_only(x, y):
        return norm(x), y

    train_ds = train_ds.map(norm_only, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(norm_only, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    model, base = build_model(num_classes)

    cb = [
        tf.keras.callbacks.EarlyStopping(
            patience=8, restore_best_weights=True, monitor="val_accuracy"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            patience=4, factor=0.5, monitor="val_loss", verbose=1
        ),
    ]

    print("[train_objects] Training (frozen backbone)...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=cb,
        verbose=1,
    )

    print("[train_objects] Fine-tuning (top MobileNet layers)...")
    base.trainable = True
    for layer in base.layers[:-24]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=min(15, EPOCHS // 2),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)
        ],
        verbose=1,
    )

    keras_path = MODEL_OUT.with_name("objects_model.keras")
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    model.save(keras_path)
    print(f"[train_objects] Keras -> {keras_path}")
    export_tflite(model)

    CLASS_NAMES_OUT.write_text(
        json.dumps({"class_names": class_names}, indent=2), encoding="utf-8"
    )
    print(f"[train_objects] Class names -> {CLASS_NAMES_OUT}")
    print("[train_objects] Done.")


def _train_kaggle():
    root = _download_kaggle_dataset()
    images_dir = _find_image_classification_root(root)
    _train_from_images_dir(images_dir)


def _train_pickle():
    import tensorflow as tf  # noqa: PLC0415
    import cv2  # noqa: PLC0415

    print("[train_objects] Loading X, y from pickle...")
    X, y, class_names, num_classes = _load_xy_pickle()
    print(f"         X {X.shape}  classes={num_classes}")

    if X.shape[1] != IMG_SIZE or X.shape[2] != IMG_SIZE:
        n, _, _, c = X.shape
        out = np.empty((n, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        for i in range(n):
            out[i] = cv2.resize(X[i], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        X = out

    X_train, y_train, X_val, y_val = _split(X, y)
    print(f"         train={len(X_train)} val={len(X_val)}")

    model, base = build_model(num_classes)

    cb = [
        tf.keras.callbacks.EarlyStopping(
            patience=8, restore_best_weights=True, monitor="val_accuracy"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            patience=4, factor=0.5, monitor="val_loss", verbose=1
        ),
    ]

    print("[train_objects] Training...")
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=cb,
        verbose=1,
    )

    print("[train_objects] Fine-tuning...")
    base.trainable = True
    for layer in base.layers[:-24]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=min(15, EPOCHS // 2),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)
        ],
        verbose=1,
    )

    keras_path = MODEL_OUT.with_name("objects_model.keras")
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    model.save(keras_path)
    print(f"[train_objects] Keras -> {keras_path}")
    export_tflite(model)

    CLASS_NAMES_OUT.write_text(
        json.dumps({"class_names": class_names}, indent=2), encoding="utf-8"
    )
    print(f"[train_objects] Class names -> {CLASS_NAMES_OUT}")
    print("[train_objects] Done.")


def main():
    p = argparse.ArgumentParser(
        description="Train objects/laptop classifier to TFLite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "For --source folder: pass the REAL path to your `train` folder (class subfolders inside).\n"
            "  Windows: in File Explorer, open that folder, click the address bar, copy path, then:\n"
            '  python train_objects.py --source folder --data-dir "C:\\Users\\You\\Downloads\\Project\\train"'
        ),
    )
    p.add_argument(
        "--source",
        choices=("kaggle", "pickle", "folder"),
        default="kaggle",
        help="kaggle = sugxm00/laptop-dataset; pickle = X.pickle/y.pickle; folder = --data-dir class subfolders",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Required for --source folder: directory with one subfolder per class (e.g. Roboflow train/)",
    )
    args = p.parse_args()

    if args.source == "kaggle":
        _train_kaggle()
    elif args.source == "pickle":
        _train_pickle()
    else:
        if args.data_dir is None:
            p.error("--data-dir is required when --source folder")
        raw_dir = str(args.data_dir)
        if _looks_like_doc_placeholder_path(raw_dir):
            p.error(
                "That --data-dir is still placeholder text (e.g. THE_PATH_YOU_COPIED), not a real folder. "
                "In File Explorer open your dataset `train` folder, click the address bar, copy the whole path, "
                "and paste only that inside the quotes. Run: python train_objects.py --help"
            )
        _train_from_images_dir(args.data_dir)


if __name__ == "__main__":
    main()
