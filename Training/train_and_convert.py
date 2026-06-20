#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import tensorflow as tf
from PIL import Image

# Keeps the original two-model design:
#   gesture  : Rock / Paper / Scissors
#   bracelet : nothing / bracelet
# Both models are trained/exported separately. The gesture model is made a little
# more shape-sensitive so Scissors is less likely to collapse into Paper.

GESTURES = ["Rock", "Paper", "Scissors"]
CONDITIONS = ["bracelet", "nothing"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

SOURCE_RESOLUTION = (128, 96)
DEFAULT_DATA_DIR = "EAI4IL-project-data"

L2 = 1e-4
LEARNING_RATE = 1e-3
NOISE_STD = 0.003
REPRESENTATIVE_SAMPLES = 200

MODELS: list[dict[str, object]] = [
    {
        "name": "gesture",
        "labels": GESTURES,
        "label_of": lambda g, c: g,
        "width": 128,
        "height": 96,
    },
    {
        "name": "bracelet",
        "labels": ["nothing", "bracelet"],
        "label_of": lambda g, c: c,
        "width": 128,
        "height": 96,
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train/export the two embedded RPS models: gesture + bracelet."
    )

    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--bmp-dir", default="data_bmp")
    parser.add_argument("--channels", type=int, default=3, choices=(1, 3))

    # Keep the original resolution unless explicitly overridden.
    parser.add_argument("--gesture-width", type=int, default=128)
    parser.add_argument("--gesture-height", type=int, default=96)
    parser.add_argument("--bracelet-width", type=int, default=128)
    parser.add_argument("--bracelet-height", type=int, default=96)

    # Small plain-Conv network. No DepthwiseConv2D: some TF/Keras versions crash
    # during TFLite conversion on depthwise ReadVariableOp.
    parser.add_argument("--gesture-base-filters", type=int, default=16)
    parser.add_argument("--bracelet-base-filters", type=int, default=12)
    parser.add_argument("--gesture-dense-units", type=int, default=72)
    parser.add_argument("--bracelet-dense-units", type=int, default=40)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)

    # Paper-vs-Scissors fix: if the model collapses Scissors into Paper, raising
    # the Scissors weight usually helps more than simply making the model huge.
    parser.add_argument("--scissors-weight", type=float, default=1.45)
    parser.add_argument("--paper-weight", type=float, default=1.00)
    parser.add_argument("--rock-weight", type=float, default=1.00)

    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument(
        "--horizontal-flip",
        action="store_true",
        help="Enable horizontal flip augmentation. Off by default because small hand datasets can be viewpoint-sensitive.",
    )

    parser.add_argument(
        "--export-raw-u8",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also export a TFLite model that accepts raw uint8 camera/BMP pixels. Recommended for embedded code.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Optional softmax temperature. Keep 1.0 while debugging accuracy.",
    )

    return parser.parse_args()


# ------------------------- dataset loading -------------------------


def discover_samples(data_dir: Path) -> list[tuple[Path, str, str, str]]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    samples: list[tuple[Path, str, str, str]] = []
    person_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir() and not p.name.startswith("."))

    for person_dir in person_dirs:
        for class_dir in sorted(c for c in person_dir.iterdir() if c.is_dir()):
            if "_" not in class_dir.name:
                continue

            gesture, condition = class_dir.name.split("_", 1)

            if gesture not in GESTURES or condition not in CONDITIONS:
                print(f"Warning: skipping unrecognized class folder {class_dir}")
                continue

            for f in sorted(class_dir.iterdir()):
                if f.suffix.lower() in IMAGE_EXTENSIONS:
                    samples.append((f, person_dir.name, gesture, condition))

    if not samples:
        raise FileNotFoundError(f"No images found under {data_dir}")

    return samples


def crop_to_aspect(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    w, h = img.size
    target_ratio = target_w / target_h
    ratio = w / h

    if ratio > target_ratio:
        new_w = int(round(h * target_ratio))
        left = (w - new_w) // 2
        return img.crop((left, 0, left + new_w, h))

    if ratio < target_ratio:
        new_h = int(round(w / target_ratio))
        top = (h - new_h) // 2
        return img.crop((0, top, w, top + new_h))

    return img


def convert_to_bmp(
    samples: list[tuple[Path, str, str, str]],
    bmp_dir: Path,
    img_width: int,
    img_height: int,
    channels: int,
) -> list[tuple[Path, str, str, str]]:
    mode = "RGB" if channels == 3 else "L"

    if bmp_dir.exists():
        shutil.rmtree(bmp_dir)

    out: list[tuple[Path, str, str, str]] = []

    for path, person, gesture, condition in samples:
        target_dir = bmp_dir / person / f"{gesture}_{condition}"
        target_dir.mkdir(parents=True, exist_ok=True)
        bmp_path = target_dir / (path.stem + ".bmp")

        with Image.open(path) as img:
            img = crop_to_aspect(img.convert(mode), img_width, img_height)
            img = img.resize((img_width, img_height), Image.BILINEAR)
            img.save(bmp_path, format="BMP")

        out.append((bmp_path, person, gesture, condition))

    print(f"Converted {len(out)} images to BMP ({img_width}x{img_height}) under {bmp_dir}")
    return out


def load_images(
    bmp_samples: list[tuple[Path, str, str, str]],
    img_width: int,
    img_height: int,
    channels: int,
) -> np.ndarray:
    mode = "RGB" if channels == 3 else "L"
    x = np.zeros((len(bmp_samples), img_height, img_width, channels), dtype=np.float32)

    for i, (path, _person, _gesture, _condition) in enumerate(bmp_samples):
        with Image.open(path) as img:
            img = crop_to_aspect(img.convert(mode), img_width, img_height)
            img = img.resize((img_width, img_height), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32)

        if channels == 1:
            arr = arr[..., None]

        x[i] = arr

    return x


def get_train_and_test_data(
    strata: np.ndarray,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for cls in np.unique(strata):
        idx = np.flatnonzero(strata == cls)
        rng.shuffle(idx)
        n = len(idx)

        if n == 1:
            train_idx.extend(idx.tolist())
            continue

        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))

        if n_test + n_val >= n:
            n_test = min(n_test, max(0, n - 1))
            n_val = max(0, n - n_test - 1)

        test_idx.extend(idx[:n_test].tolist())
        val_idx.extend(idx[n_test:n_test + n_val].tolist())
        train_idx.extend(idx[n_test + n_val:].tolist())

    for idx in (train_idx, val_idx, test_idx):
        rng.shuffle(idx)

    return (
        np.asarray(train_idx, dtype=np.int64),
        np.asarray(val_idx, dtype=np.int64),
        np.asarray(test_idx, dtype=np.int64),
    )


def fit_normalizer(x_train_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = x_train_raw.reshape(-1, x_train_raw.shape[-1])
    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def apply_normalizer(x_raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x_raw - mean.reshape(1, 1, 1, -1)) / std.reshape(1, 1, 1, -1)).astype(np.float32)


# ------------------------- augmentation / datasets -------------------------


def make_augmenter(horizontal_flip: bool) -> tf.keras.Sequential:
    layers: list[tf.keras.layers.Layer] = []

    if horizontal_flip:
        layers.append(tf.keras.layers.RandomFlip("horizontal"))

    # Keep augmentation mild. Large rotations/zooms make Paper and Scissors more
    # similar on a tiny dataset.
    layers.extend([
        tf.keras.layers.RandomRotation(0.05, fill_mode="reflect"),
        tf.keras.layers.RandomTranslation(0.04, 0.04, fill_mode="reflect"),
        tf.keras.layers.RandomZoom(0.04, fill_mode="reflect"),
        tf.keras.layers.RandomContrast(0.12),
    ])

    return tf.keras.Sequential(layers, name="augmenter")


def make_dataset(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    training: bool,
    seed: int,
    augment: bool,
    augmenter: tf.keras.Sequential | None,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((x.astype(np.float32), y.astype(np.int64)))

    if training:
        ds = ds.shuffle(max(1, len(x)), seed=seed, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size)

    if training and augment and augmenter is not None:
        def aug_fn(a, b):
            a = augmenter(a, training=True)
            a = a + tf.random.normal(tf.shape(a), mean=0.0, stddev=NOISE_STD, dtype=a.dtype)
            return a, b

        ds = ds.map(aug_fn, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.prefetch(tf.data.AUTOTUNE)


# ------------------------- model -------------------------


def conv_bn_relu(x: tf.Tensor, filters: int, kernel_size: int, name: str) -> tf.Tensor:
    reg = tf.keras.regularizers.l2(L2) if L2 > 0.0 else None
    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size,
        padding="same",
        use_bias=False,
        kernel_regularizer=reg,
        name=f"{name}_conv",
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn")(x)
    x = tf.keras.layers.ReLU(name=f"{name}_relu")(x)
    return x


def make_model(
    img_width: int,
    img_height: int,
    channels: int,
    num_classes: int,
    name: str,
    base_filters: int,
    dense_units: int,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(img_height, img_width, channels), name="image")

    # Plain Conv2D only: converter-stable and still small.
    x = conv_bn_relu(inputs, base_filters, 5, "block1a")
    x = conv_bn_relu(x, base_filters, 3, "block1b")
    x = tf.keras.layers.MaxPooling2D(pool_size=2, name="pool1")(x)
    x = tf.keras.layers.Dropout(0.05, name="drop1")(x)

    x = conv_bn_relu(x, base_filters * 2, 3, "block2a")
    x = conv_bn_relu(x, base_filters * 2, 3, "block2b")
    x = tf.keras.layers.MaxPooling2D(pool_size=2, name="pool2")(x)
    x = tf.keras.layers.Dropout(0.08, name="drop2")(x)

    x = conv_bn_relu(x, base_filters * 3, 3, "block3a")
    x = conv_bn_relu(x, base_filters * 3, 3, "block3b")
    x = tf.keras.layers.MaxPooling2D(pool_size=2, name="pool3")(x)

    # Important Paper-vs-Scissors change:
    # average pooling alone says "how much open-hand feature exists".
    # max pooling preserves stronger local evidence, e.g. two separated fingers.
    avg = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    mx = tf.keras.layers.GlobalMaxPooling2D(name="gmp")(x)
    x = tf.keras.layers.Concatenate(name="gap_gmp")([avg, mx])

    x = tf.keras.layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(L2) if L2 > 0.0 else None,
        name="hidden",
    )(x)
    x = tf.keras.layers.Dropout(0.12, name="hidden_drop")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def make_temperature_model(model: tf.keras.Model, temperature: float) -> tf.keras.Model:
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")
    if abs(temperature - 1.0) < 1e-8:
        return model

    probs = model.output
    clipped = tf.keras.layers.Lambda(lambda p: tf.clip_by_value(p, 1e-7, 1.0), name="clip_probs")(probs)
    log_probs = tf.keras.layers.Lambda(lambda p: tf.math.log(p), name="log_probs")(clipped)
    scaled = tf.keras.layers.Rescaling(1.0 / temperature, name="temperature_scale")(log_probs)
    sharpened = tf.keras.layers.Activation("softmax", name="temperature_softmax")(scaled)
    return tf.keras.Model(model.input, sharpened, name=f"{model.name}_temperature")


def make_raw_input_model(
    normalized_model: tf.keras.Model,
    mean: np.ndarray,
    std: np.ndarray,
    img_width: int,
    img_height: int,
    channels: int,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(img_height, img_width, channels), name="image_raw")
    norm = tf.keras.layers.Normalization(
        axis=-1,
        mean=mean.astype(np.float32),
        variance=np.square(std.astype(np.float32)),
        name="zscore_from_raw_pixels",
    )
    x = norm(inputs)
    outputs = normalized_model(x)
    return tf.keras.Model(inputs, outputs, name=f"{normalized_model.name}_raw_input")


# ------------------------- metrics -------------------------


def compute_class_weights(y: np.ndarray, labels: list[str], model_name: str, args) -> dict[int, float]:
    num_classes = len(labels)
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    total = float(np.sum(counts))

    weights: dict[int, float] = {}
    for cls, count in enumerate(counts):
        if count > 0:
            weights[cls] = total / (num_classes * float(count))

    if model_name == "gesture":
        boosts = {
            "Rock": args.rock_weight,
            "Paper": args.paper_weight,
            "Scissors": args.scissors_weight,
        }
        for i, label in enumerate(labels):
            if i in weights:
                weights[i] *= float(boosts.get(label, 1.0))

    # Normalize so the average weight stays near 1.0.
    if weights:
        avg = float(np.mean(list(weights.values())))
        if avg > 0:
            weights = {k: v / avg for k, v in weights.items()}

    return weights


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def accuracy_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    if len(y_true) == 0:
        return float("nan"), float("nan")
    acc = float(np.mean(y_true == y_pred))
    recalls = []
    for cls in np.unique(y_true):
        mask = y_true == cls
        recalls.append(float(np.mean(y_pred[mask] == cls)))
    bal = float(np.mean(recalls)) if recalls else float("nan")
    return acc, bal


def evaluate_keras(model: tf.keras.Model, x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
    if len(x) == 0:
        return float("nan"), float("nan"), np.array([], dtype=np.int64), np.empty((0, 0), dtype=np.int64)
    probs = model.predict(x, verbose=0)
    pred = np.argmax(probs, axis=1).astype(np.int64)
    acc, bal = accuracy_from_preds(y, pred)
    cm = confusion_matrix_np(y, pred, probs.shape[1])
    return acc, bal, pred, cm


def save_confusion_csv(path: Path, cm: np.ndarray, labels: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["actual\\predicted", *labels])
        for i, label in enumerate(labels):
            writer.writerow([label, *cm[i].tolist()])


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def count_parameters(model: tf.keras.Model) -> int:
    return int(np.sum([np.prod(w.shape) for w in model.weights]))


def file_kb(path: Path) -> float:
    return path.stat().st_size / 1024.0


# ------------------------- TFLite -------------------------


def representative_dataset(x: np.ndarray) -> Iterable[list[np.ndarray]]:
    if len(x) == 0:
        return
    n = min(len(x), REPRESENTATIVE_SAMPLES)
    indices = np.linspace(0, len(x) - 1, num=n, dtype=np.int64)
    for idx in indices:
        yield [x[idx:idx + 1].astype(np.float32)]


def export_saved_model(model: tf.keras.Model, saved_model_dir: Path) -> None:
    if saved_model_dir.exists():
        shutil.rmtree(saved_model_dir)
    saved_model_dir.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "export"):
        model.export(str(saved_model_dir))
    else:
        tf.saved_model.save(model, str(saved_model_dir))


def convert_tflite_float(model: tf.keras.Model, out_path: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        saved_model_dir = Path(tmp) / "saved_model"
        export_saved_model(model, saved_model_dir)
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        out_path.write_bytes(converter.convert())


def convert_tflite_int(
    model: tf.keras.Model,
    representative_x: np.ndarray,
    out_path: Path,
    input_type: tf.dtypes.DType,
    output_type: tf.dtypes.DType,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        saved_model_dir = Path(tmp) / "saved_model"
        export_saved_model(model, saved_model_dir)
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: representative_dataset(representative_x)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = input_type
        converter.inference_output_type = output_type
        out_path.write_bytes(converter.convert())


def set_tflite_input(interpreter: tf.lite.Interpreter, x_one: np.ndarray) -> None:
    input_details = interpreter.get_input_details()[0]
    value = x_one.astype(np.float32)

    if input_details["dtype"] in (np.int8, np.uint8):
        scale, zero_point = input_details["quantization"]
        if scale == 0:
            raise ValueError("Quantized model input scale is 0")
        value = np.round(value / scale + zero_point)
        qinfo = np.iinfo(input_details["dtype"])
        value = np.clip(value, qinfo.min, qinfo.max).astype(input_details["dtype"])
    else:
        value = value.astype(input_details["dtype"])

    interpreter.set_tensor(input_details["index"], value)


def get_tflite_output(interpreter: tf.lite.Interpreter) -> np.ndarray:
    output_details = interpreter.get_output_details()[0]
    value = interpreter.get_tensor(output_details["index"])

    if output_details["dtype"] in (np.int8, np.uint8):
        scale, zero_point = output_details["quantization"]
        value = (value.astype(np.float32) - float(zero_point)) * float(scale)

    return value.astype(np.float32)


def evaluate_tflite(path: Path, x: np.ndarray, y: np.ndarray, num_classes: int) -> tuple[float, float, float, np.ndarray]:
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan"), np.zeros((num_classes, num_classes), dtype=np.int64)

    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()

    preds: list[int] = []
    confidences: list[float] = []

    for i in range(len(x)):
        set_tflite_input(interpreter, x[i:i + 1])
        interpreter.invoke()
        probs = get_tflite_output(interpreter)[0]
        preds.append(int(np.argmax(probs)))
        confidences.append(float(np.max(probs)))

    pred_arr = np.asarray(preds, dtype=np.int64)
    acc, bal = accuracy_from_preds(y, pred_arr)
    cm = confusion_matrix_np(y, pred_arr, num_classes)
    avg_conf = float(np.mean(confidences)) if confidences else float("nan")
    return acc, bal, avg_conf, cm


# ------------------------- training/export -------------------------


def train_and_export(
    spec: dict,
    x_raw: np.ndarray,
    x_norm: np.ndarray,
    gestures: list[str],
    conditions: list[str],
    split: tuple[np.ndarray, np.ndarray, np.ndarray],
    args,
    model_dir: Path,
    mean: np.ndarray,
    std: np.ndarray,
    img_width: int,
    img_height: int,
) -> list[dict]:
    name = str(spec["name"])
    labels: list[str] = spec["labels"]  # type: ignore[assignment]
    label_of: Callable[[str, str], str] = spec["label_of"]  # type: ignore[assignment]
    num_classes = len(labels)

    y = np.asarray([labels.index(label_of(g, c)) for g, c in zip(gestures, conditions)], dtype=np.int64)
    train_idx, val_idx, test_idx = split

    x_train = x_norm[train_idx]
    x_val = x_norm[val_idx]
    x_test = x_norm[test_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    x_train_raw = x_raw[train_idx]
    x_test_raw = x_raw[test_idx]

    base_filters = args.gesture_base_filters if name == "gesture" else args.bracelet_base_filters
    dense_units = args.gesture_dense_units if name == "gesture" else args.bracelet_dense_units

    print(f"\n=== {name} ({num_classes}-class, {img_width}x{img_height}, channels={args.channels}) ===")
    print(f"labels={labels}")
    print(f"train class counts: {np.bincount(y_train, minlength=num_classes).tolist()}")
    if name == "gesture":
        print(
            "gesture class weight boosts: "
            f"Rock={args.rock_weight}, Paper={args.paper_weight}, Scissors={args.scissors_weight}"
        )

    augmenter = make_augmenter(args.horizontal_flip)
    train_ds = make_dataset(
        x_train,
        y_train,
        args.batch_size,
        training=True,
        seed=args.seed,
        augment=not args.no_augment,
        augmenter=augmenter,
    )
    val_ds = make_dataset(x_val, y_val, args.batch_size, False, args.seed, False, None) if len(x_val) else None

    model = make_model(
        img_width,
        img_height,
        args.channels,
        num_classes,
        name,
        base_filters,
        dense_units,
    )

    class_weight = compute_class_weights(y_train, labels, name, args)
    print(f"class weights: {class_weight}")

    monitor = "val_accuracy" if val_ds is not None else "accuracy"
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            mode="max",
            factor=0.5,
            patience=12,
            min_lr=1e-5,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode="max",
            patience=35,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        verbose=2,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    # Evaluate unquantized Keras model.
    rows: list[dict] = []
    for split_name, x_eval, y_eval in [
        ("train", x_train, y_train),
        ("val", x_val, y_val),
        ("test", x_test, y_test),
    ]:
        acc, bal, _pred, cm = evaluate_keras(model, x_eval, y_eval)
        print(f"{split_name:>5} keras accuracy={acc:.4f}, balanced_accuracy={bal:.4f}")
        save_confusion_csv(model_dir / f"confusion_{split_name}_keras.csv", cm, labels)
        rows.append({
            "model": name,
            "variant": f"keras_{split_name}",
            "file": "",
            "accuracy": f"{acc:.6f}",
            "balanced_accuracy": f"{bal:.6f}",
            "parameters": count_parameters(model),
            "size_kb": "",
            "avg_confidence": "",
            "notes": "Keras float model; input is externally normalized x=(px-mean)/std",
        })

    save_json(model_dir / "label_map.json", {
        "id_to_label": {str(i): label for i, label in enumerate(labels)},
        "label_to_id": {label: i for i, label in enumerate(labels)},
    })

    save_json(model_dir / "normalization.json", {
        "format": "bmp",
        "color_mode": "RGB" if args.channels == 3 else "grayscale",
        "source_resolution": list(SOURCE_RESOLUTION),
        "image_width": img_width,
        "image_height": img_height,
        "channels": args.channels,
        "input_shape": [1, img_height, img_width, args.channels],
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "preprocessing_for_model_i8": (
            "Use this if your embedded code already performs z-score normalization: "
            "center-crop to model aspect, resize bilinear, convert color mode, "
            "normalize raw 0..255 pixels using (px-mean)/std, then quantize using the TFLite input scale/zero_point."
        ),
        "preprocessing_for_model_raw_u8": (
            "Recommended for embedded code: center-crop to model aspect, resize bilinear, "
            "convert color mode, and feed raw uint8 pixels 0..255 directly. The z-score normalization is inside the model."
        ),
    })

    deployment_model = make_temperature_model(model, args.temperature)

    # Backward-compatible normalized-input exports.
    float_path = model_dir / "model_f32.tflite"
    i8_path = model_dir / "model_i8.tflite"

    convert_tflite_float(deployment_model, float_path)
    convert_tflite_int(deployment_model, x_train, i8_path, tf.int8, tf.int8)

    for variant_name, path, x_eval in [
        ("model_f32", float_path, x_test),
        ("model_i8", i8_path, x_test),
    ]:
        acc, bal, conf, cm = evaluate_tflite(path, x_eval, y_test, num_classes)
        save_confusion_csv(model_dir / f"confusion_test_{variant_name}.csv", cm, labels)
        print(f"  exported {path.name:<20} test_acc={acc:.4f} bal_acc={bal:.4f} size={file_kb(path):6.1f} KB")
        rows.append({
            "model": name,
            "variant": variant_name,
            "file": f"{name}/{path.name}",
            "accuracy": f"{acc:.6f}",
            "balanced_accuracy": f"{bal:.6f}",
            "parameters": count_parameters(model),
            "size_kb": f"{file_kb(path):.1f}",
            "avg_confidence": f"{conf:.6f}",
            "notes": "TFLite; normalized input expected",
        })

    # Recommended embedded export: raw uint8 input, internal z-score normalization.
    if args.export_raw_u8:
        raw_model = make_raw_input_model(deployment_model, mean, std, img_width, img_height, args.channels)
        raw_u8_path = model_dir / "model_raw_u8.tflite"
        convert_tflite_int(raw_model, x_train_raw, raw_u8_path, tf.uint8, tf.uint8)

        acc, bal, conf, cm = evaluate_tflite(raw_u8_path, x_test_raw, y_test, num_classes)
        save_confusion_csv(model_dir / "confusion_test_model_raw_u8.csv", cm, labels)
        print(f"  exported {raw_u8_path.name:<20} test_acc={acc:.4f} bal_acc={bal:.4f} size={file_kb(raw_u8_path):6.1f} KB")
        rows.append({
            "model": name,
            "variant": "model_raw_u8",
            "file": f"{name}/{raw_u8_path.name}",
            "accuracy": f"{acc:.6f}",
            "balanced_accuracy": f"{bal:.6f}",
            "parameters": count_parameters(raw_model),
            "size_kb": f"{file_kb(raw_u8_path):.1f}",
            "avg_confidence": f"{conf:.6f}",
            "notes": "Recommended TFLite; raw uint8 0..255 image input; normalization inside model",
        })

    return rows


def main() -> int:
    args = parse_args()

    if args.temperature != 1.0:
        print("Warning: use --temperature 1.0 while debugging Paper/Scissors confusion.")

    tf.keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    samples = discover_samples(Path(args.data_dir))
    gestures = [g for _p, _person, g, _c in samples]
    conditions = [c for _p, _person, _g, c in samples]

    combined = [f"{g}_{c}" for g, c in zip(gestures, conditions)]
    strata_names = sorted(set(combined))
    strata = np.asarray([strata_names.index(s) for s in combined], dtype=np.int64)

    split = get_train_and_test_data(strata, args.val_ratio, args.test_ratio, args.seed)
    train_idx, val_idx, test_idx = split

    print(f"Discovered {len(samples)} images from {args.data_dir}")
    print(f"Strata: {dict(enumerate(strata_names))}")
    print(f"Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    sizes = {
        "gesture": (args.gesture_width, args.gesture_height),
        "bracelet": (args.bracelet_width, args.bracelet_height),
    }

    all_rows: list[dict] = []

    for spec in MODELS:
        name = str(spec["name"])
        width, height = sizes[name]
        model_dir = artifacts_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)

        bmp_samples = convert_to_bmp(samples, Path(args.bmp_dir) / name, width, height, args.channels)
        x_raw = load_images(bmp_samples, width, height, args.channels)
        mean, std = fit_normalizer(x_raw[train_idx])
        x_norm = apply_normalizer(x_raw, mean, std)

        all_rows.extend(train_and_export(
            spec=spec,
            x_raw=x_raw,
            x_norm=x_norm,
            gestures=gestures,
            conditions=conditions,
            split=split,
            args=args,
            model_dir=model_dir,
            mean=mean,
            std=std,
            img_width=width,
            img_height=height,
        ))

    metrics_path = artifacts_dir / "model_metrics.csv"
    if all_rows:
        with metrics_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

    print()
    print(f"Saved metrics: {metrics_path}")
    print(f"Models are under: {artifacts_dir}/gesture and {artifacts_dir}/bracelet")
    print("For embedded use, prefer model_raw_u8.tflite unless your C++ already performs z-score normalization.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
