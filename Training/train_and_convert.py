#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import tensorflow as tf
from PIL import Image


GESTURES = ["Rock", "Paper", "Scissors"]
CONDITIONS = ["bracelet", "nothing"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

SOURCE_RESOLUTION = (640, 480)

# Both models now use the same input image size.
DEFAULT_MODEL_WIDTH = 128
DEFAULT_MODEL_HEIGHT = 96

BASE_FILTERS = 24
DENSE_UNITS = 128
DROPOUT = 0.20
L2 = 1e-5
LEARNING_RATE = 5e-4
NOISE_STD = 0.01
REPRESENTATIVE_SAMPLES = 200

# Keep this at 1.0 while debugging. Lower values only sharpen confidence;
# they do not improve a weak model.
SOFTMAX_TEMPERATURE = 1.0

MODELS: list[dict[str, object]] = [
    {
        "name": "gesture",
        "labels": GESTURES,
        "label_of": lambda g, c: g,
        "width": DEFAULT_MODEL_WIDTH,
        "height": DEFAULT_MODEL_HEIGHT,
    },
    {
        "name": "bracelet",
        "labels": ["nothing", "bracelet"],
        "label_of": lambda g, c: c,
        "width": DEFAULT_MODEL_WIDTH,
        "height": DEFAULT_MODEL_HEIGHT,
    },
]

DEFAULT_DATA_DIR = "EAI4IL-project-data"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the RPS gesture + bracelet image models and export TFLite."
    )

    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--bmp-dir", default="data_bmp")
    parser.add_argument("--channels", type=int, default=3, choices=(1, 3))

    parser.add_argument("--gesture-width", type=int, default=DEFAULT_MODEL_WIDTH)
    parser.add_argument("--gesture-height", type=int, default=DEFAULT_MODEL_HEIGHT)
    parser.add_argument("--bracelet-width", type=int, default=DEFAULT_MODEL_WIDTH)
    parser.add_argument("--bracelet-height", type=int, default=DEFAULT_MODEL_HEIGHT)

    parser.add_argument("--base-filters", type=int, default=BASE_FILTERS)
    parser.add_argument("--dense-units", type=int, default=DENSE_UNITS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--no-augment", action="store_true")

    parser.add_argument(
        "--temperature",
        type=float,
        default=SOFTMAX_TEMPERATURE,
        help="Use 1.0 while debugging. Lower values only sharpen output probabilities.",
    )

    parser.add_argument("--prune", action="store_true")
    parser.add_argument("--synapse-prune-ratio", type=float, default=0.50)
    parser.add_argument("--neuron-prune-ratio", type=float, default=0.25)

    return parser.parse_args()


def discover_samples(data_dir: Path) -> list[tuple[Path, str, str, str]]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    samples: list[tuple[Path, str, str, str]] = []

    person_dirs = sorted(
        p for p in data_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )

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
            img = img.resize((img_width, img_height), Image.Resampling.BILINEAR)
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
            arr = np.asarray(
                img.resize((img_width, img_height), Image.Resampling.BILINEAR),
                dtype=np.float32,
            )

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


def fit_normalizer(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = x_train.reshape(-1, x_train.shape[-1])

    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)

    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    return mean, std


def apply_normalizer(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean.reshape(1, 1, 1, -1)) /
            std.reshape(1, 1, 1, -1)).astype(np.float32)


def compute_class_weights(y: np.ndarray, num_classes: int) -> dict[int, float]:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    total = float(np.sum(counts))

    return {
        cls: total / (num_classes * float(count))
        for cls, count in enumerate(counts)
        if count > 0
    }


# Less destructive augmentation. Horizontal flip is intentionally removed.
_AUGMENTER = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.03, fill_mode="reflect"),
    tf.keras.layers.RandomZoom(0.05, fill_mode="reflect"),
    tf.keras.layers.RandomTranslation(0.04, 0.04, fill_mode="reflect"),
])


def augment_batch(x: tf.Tensor, y: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    x = _AUGMENTER(x, training=True)

    if NOISE_STD > 0.0:
        x = x + tf.random.normal(
            tf.shape(x),
            mean=0.0,
            stddev=NOISE_STD,
            dtype=x.dtype,
        )

    return x, y


def make_dataset(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    training: bool,
    seed: int,
    augment: bool = False,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(
        (x.astype(np.float32), y.astype(np.int64))
    )

    if training:
        ds = ds.shuffle(
            buffer_size=max(1, len(x)),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    ds = ds.batch(batch_size)

    if training and augment:
        ds = ds.map(augment_batch, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.prefetch(tf.data.AUTOTUNE)


def conv_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: int = 3,
    pool: bool = True,
    dropout: float = 0.0,
) -> tf.Tensor:
    reg = tf.keras.regularizers.l2(L2) if L2 > 0.0 else None

    shortcut = x

    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size,
        padding="same",
        use_bias=False,
        kernel_regularizer=reg,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
        kernel_regularizer=reg,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    if shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(
            filters,
            1,
            padding="same",
            use_bias=False,
            kernel_regularizer=reg,
        )(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)

    if pool:
        x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)

    if dropout > 0.0:
        x = tf.keras.layers.SpatialDropout2D(dropout)(x)

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
    inputs = tf.keras.Input(
        shape=(img_height, img_width, channels),
        name="image",
    )

    x = conv_block(
        inputs,
        base_filters,
        kernel_size=5,
        pool=True,
        dropout=0.05,
    )

    x = conv_block(
        x,
        base_filters * 2,
        kernel_size=3,
        pool=True,
        dropout=0.10,
    )

    x = conv_block(
        x,
        base_filters * 4,
        kernel_size=3,
        pool=True,
        dropout=0.10,
    )

    # One extra block without pooling keeps the remaining spatial pattern.
    x = conv_block(
        x,
        base_filters * 4,
        kernel_size=3,
        pool=False,
        dropout=0.10,
    )

    # Do not use GlobalAveragePooling here; it can erase shape/position cues.
    x = tf.keras.layers.Conv2D(
        base_filters * 4,
        3,
        padding="same",
        activation="relu",
        name="final_spatial_conv",
    )(x)

    x = tf.keras.layers.Flatten(name="flatten_spatial_features")(x)

    x = tf.keras.layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(L2) if L2 > 0.0 else None,
        name="hidden",
    )(x)

    if DROPOUT > 0.0:
        x = tf.keras.layers.Dropout(DROPOUT)(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="output",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        ],
    )

    return model


def make_confident_deployment_model(
    trained_softmax_model: tf.keras.Model,
    temperature: float,
) -> tf.keras.Model:
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")

    if abs(temperature - 1.0) < 1e-6:
        return trained_softmax_model

    probs = trained_softmax_model.output

    clipped = tf.keras.layers.Lambda(
        lambda p: tf.clip_by_value(p, 1e-7, 1.0),
        name="clip_probs",
    )(probs)

    log_probs = tf.keras.layers.Lambda(
        lambda p: tf.math.log(p),
        name="log_probs",
    )(clipped)

    scaled_log_probs = tf.keras.layers.Rescaling(
        scale=1.0 / temperature,
        name="temperature_scale",
    )(log_probs)

    sharpened_probs = tf.keras.layers.Activation(
        "softmax",
        name="confident_output",
    )(scaled_log_probs)

    return tf.keras.Model(
        inputs=trained_softmax_model.input,
        outputs=sharpened_probs,
        name=f"{trained_softmax_model.name}_confident",
    )


def prune_synapses(base_model: tf.keras.Model, prune_ratio: float) -> tf.keras.Model:
    if not 0.0 <= prune_ratio < 1.0:
        raise ValueError("prune_ratio must satisfy 0.0 <= prune_ratio < 1.0")

    model = tf.keras.models.clone_model(base_model)
    model.set_weights(base_model.get_weights())

    kernels: list[np.ndarray] = []

    for layer in model.layers:
        if hasattr(layer, "kernel"):
            kernels.append(np.abs(layer.kernel.numpy()).reshape(-1))

    if not kernels:
        raise ValueError("No prunable kernels found.")

    all_kernel_values = np.concatenate(kernels)
    threshold = np.percentile(all_kernel_values, prune_ratio * 100.0)

    for layer in model.layers:
        if not hasattr(layer, "kernel"):
            continue

        weights = layer.get_weights()
        kernel = weights[0]
        kernel[np.abs(kernel) < threshold] = 0.0
        weights[0] = kernel
        layer.set_weights(weights)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    return model


def keep_count(total: int, prune_ratio: float) -> int:
    if not 0.0 <= prune_ratio <= 0.95:
        raise ValueError("prune_ratio must satisfy 0.0 <= prune_ratio <= 0.95")

    return max(1, int(round(total * (1.0 - prune_ratio))))


def prune_neurons(base_model: tf.keras.Model, prune_ratio: float) -> tf.keras.Model:
    model = tf.keras.models.clone_model(base_model)
    model.set_weights(base_model.get_weights())

    hidden = model.get_layer("hidden")
    output = model.get_layer("output")

    feature_extractor = tf.keras.Model(
        inputs=model.input,
        outputs=hidden.input,
        name="feature_extractor",
    )

    hidden_kernel, hidden_bias = hidden.get_weights()
    output_kernel, output_bias = output.get_weights()

    old_units = hidden_kernel.shape[1]
    new_units = keep_count(old_units, prune_ratio)

    incoming_scores = np.sum(np.abs(hidden_kernel), axis=0)
    outgoing_scores = np.sum(np.abs(output_kernel), axis=1)
    scores = incoming_scores + outgoing_scores

    keep_indices = np.sort(np.argsort(scores)[-new_units:])

    inputs = tf.keras.Input(shape=base_model.input_shape[1:], name="image")
    x = feature_extractor(inputs)

    x = tf.keras.layers.Dense(
        new_units,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(L2) if L2 > 0.0 else None,
        name="hidden",
    )(x)

    if DROPOUT > 0.0:
        x = tf.keras.layers.Dropout(DROPOUT)(x)

    outputs = tf.keras.layers.Dense(
        output_kernel.shape[1],
        activation="softmax",
        name="output",
    )(x)

    pruned_model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=f"{base_model.name}_neuron_pruned",
    )

    pruned_model.get_layer("hidden").set_weights([
        hidden_kernel[:, keep_indices],
        hidden_bias[keep_indices],
    ])

    pruned_model.get_layer("output").set_weights([
        output_kernel[keep_indices, :],
        output_bias,
    ])

    pruned_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    return pruned_model


def export_saved_model(model: tf.keras.Model, saved_model_dir: Path):
    if saved_model_dir.exists():
        shutil.rmtree(saved_model_dir)

    saved_model_dir.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "export"):
        model.export(str(saved_model_dir))
    else:
        tf.saved_model.save(model, str(saved_model_dir))


def representative_dataset(x_train: np.ndarray) -> Iterable[list[np.ndarray]]:
    n = min(len(x_train), REPRESENTATIVE_SAMPLES)
    indices = np.linspace(0, len(x_train) - 1, num=n, dtype=np.int64)

    for idx in indices:
        yield [x_train[idx:idx + 1].astype(np.float32)]


def export_tflite_models(
    model: tf.keras.Model,
    model_dir: Path,
    x_train: np.ndarray,
    file_stem: str,
) -> dict[str, Path]:
    saved_model_dir = model_dir / "saved_models" / file_stem
    export_saved_model(model, saved_model_dir)

    outputs: dict[str, Path] = {}

    float_converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    float_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    float_path = model_dir / f"{file_stem}_f32.tflite"
    float_path.write_bytes(float_converter.convert())
    outputs["float32"] = float_path

    int8_converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    int8_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    int8_converter.representative_dataset = lambda: representative_dataset(x_train)
    int8_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    int8_converter.inference_input_type = tf.int8
    int8_converter.inference_output_type = tf.int8
    int8_path = model_dir / f"{file_stem}_i8.tflite"

    try:
        int8_path.write_bytes(int8_converter.convert())
        outputs["int8"] = int8_path
    except Exception as exc:
        print(f"Warning: int8 TFLite conversion failed for {file_stem}: {exc}")

    return outputs


def set_tflite_input(interpreter: tf.lite.Interpreter, x_one: np.ndarray):
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

    return value


def evaluate_tflite(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    num_classes: int,
) -> tuple[float, float, float, list[int], list[int]]:
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

    preds_arr = np.asarray(preds, dtype=np.int64)

    accuracy = float((preds_arr == y).mean()) if len(x) else float("nan")

    recalls: list[float] = []
    for cls in range(num_classes):
        mask = y == cls
        if np.any(mask):
            recalls.append(float((preds_arr[mask] == cls).mean()))

    balanced = float(np.mean(recalls)) if recalls else float("nan")
    avg_confidence = float(np.mean(confidences)) if confidences else float("nan")

    pred_counts = np.bincount(preds_arr, minlength=num_classes).tolist()
    true_counts = np.bincount(y, minlength=num_classes).tolist()

    return accuracy, balanced, avg_confidence, pred_counts, true_counts


def count_parameters(model: tf.keras.Model) -> int:
    return int(np.sum([np.prod(w.shape) for w in model.weights]))


def file_kb(path: Path) -> float:
    return path.stat().st_size / 1024.0


def save_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def cpp_array(values: np.ndarray) -> str:
    return "{ " + ", ".join(f"{float(v):.9g}F" for v in values) + " }"


def save_preprocessing_cpp_header(
    artifacts_dir: Path,
    configs: dict[str, dict[str, object]],
):
    lines: list[str] = [
        "#pragma once",
        "",
        "#include <array>",
        "",
        "namespace preprocessing {",
        "",
    ]

    for name, cfg in configs.items():
        prefix = name.upper()

        lines.extend([
            f"inline constexpr int {prefix}_IMG_WIDTH = {int(cfg['width'])};",
            f"inline constexpr int {prefix}_IMG_HEIGHT = {int(cfg['height'])};",
            f"inline constexpr int {prefix}_IMG_CHANNELS = {int(cfg['channels'])};",
            f"inline constexpr int {prefix}_NUM_CLASSES = {int(cfg['num_classes'])};",
            f"inline constexpr std::array<float, 3> {prefix}_MEAN = {cpp_array(np.asarray(cfg['mean'], dtype=np.float32))};",
            f"inline constexpr std::array<float, 3> {prefix}_STD = {cpp_array(np.asarray(cfg['std'], dtype=np.float32))};",
            "",
        ])

    lines.append("}  // namespace preprocessing")
    lines.append("")

    header_path = artifacts_dir / "preprocessing_config.h"
    header_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved C++ preprocessing header: {header_path}")


def train_and_export(
    spec: dict,
    x: np.ndarray,
    gestures: list[str],
    conditions: list[str],
    split: tuple[np.ndarray, np.ndarray, np.ndarray],
    args,
    model_dir: Path,
    img_width: int,
    img_height: int,
) -> tuple[list[dict], int]:
    name = str(spec["name"])
    labels: list[str] = spec["labels"]  # type: ignore[assignment]
    label_of: Callable[[str, str], str] = spec["label_of"]  # type: ignore[assignment]
    num_classes = len(labels)

    y = np.asarray(
        [labels.index(label_of(g, c)) for g, c in zip(gestures, conditions)],
        dtype=np.int64,
    )

    train_idx, val_idx, test_idx = split
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    print(f"\n=== {name} ({num_classes}-class, {img_width}x{img_height}) ===")
    print(f"  train class counts: {np.bincount(y_train, minlength=num_classes).tolist()}  labels={labels}")

    train_ds = make_dataset(
        x_train,
        y_train,
        args.batch_size,
        training=True,
        seed=args.seed,
        augment=not args.no_augment,
    )
    val_ds = make_dataset(x_val, y_val, args.batch_size, False, args.seed) if len(x_val) else None

    model = make_model(
        img_width,
        img_height,
        args.channels,
        num_classes,
        name,
        args.base_filters,
        args.dense_units,
    )

    monitor = "val_accuracy" if val_ds is not None else "accuracy"

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            mode="max",
            factor=0.5,
            patience=10,
            min_lr=1e-5,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode="max",
            patience=30,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        verbose=2,
        class_weight=compute_class_weights(y_train, num_classes),
        callbacks=callbacks,
    )

    save_json(model_dir / "label_map.json", {
        "id_to_label": {str(i): label for i, label in enumerate(labels)},
        "label_to_id": {label: i for i, label in enumerate(labels)},
    })

    variants = [("model", model, "original")]

    if args.prune:
        variants.append((
            "model_synapse_pruned",
            prune_synapses(model, args.synapse_prune_ratio),
            f"synapse prune ratio={args.synapse_prune_ratio}",
        ))
        variants.append((
            "model_neuron_pruned",
            prune_neurons(model, args.neuron_prune_ratio),
            f"neuron prune ratio={args.neuron_prune_ratio}",
        ))

    rows: list[dict] = []

    for file_stem, base, note in variants:
        deployment_model = make_confident_deployment_model(base, args.temperature)
        tflite_paths = export_tflite_models(
            deployment_model,
            model_dir,
            x_train,
            file_stem,
        )

        for kind, path in tflite_paths.items():
            if len(x_test):
                acc, bal, conf, pred_counts, true_counts = evaluate_tflite(
                    path,
                    x_test,
                    y_test,
                    num_classes,
                )
            else:
                acc = bal = conf = float("nan")
                pred_counts = [0] * num_classes
                true_counts = [0] * num_classes

            rows.append({
                "model": name,
                "variant": f"{file_stem}_{kind}",
                "file": f"{name}/{path.name}",
                "accuracy": f"{acc:.6f}",
                "balanced_accuracy": f"{bal:.6f}",
                "parameters": count_parameters(base),
                "size_kb": f"{file_kb(path):.1f}",
                "prediction_counts": str(pred_counts),
                "true_counts": str(true_counts),
                "notes": f"{note}, {kind}, temperature={args.temperature}, avg_confidence={conf:.4f}",
            })

            print(
                f"  exported {path.name:<26} "
                f"acc={acc:.4f} bal_acc={bal:.4f} "
                f"pred={pred_counts} true={true_counts} "
                f"size={file_kb(path):6.1f} KB"
            )

    return rows, num_classes


def main():
    args = parse_args()

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

    split = get_train_and_test_data(
        strata,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )
    train_idx, val_idx, test_idx = split

    print(f"Discovered {len(samples)} images from {args.data_dir}")
    print(f"Strata: {dict(enumerate(strata_names))}")
    print(f"Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    sizes = {
        "gesture": (args.gesture_width, args.gesture_height),
        "bracelet": (args.bracelet_width, args.bracelet_height),
    }

    rows: list[dict] = []
    preprocessing_configs: dict[str, dict[str, object]] = {}

    for spec in MODELS:
        name = str(spec["name"])
        width, height = sizes[name]
        model_dir = artifacts_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)

        bmp_samples = convert_to_bmp(
            samples,
            Path(args.bmp_dir) / name,
            width,
            height,
            args.channels,
        )

        x_raw = load_images(bmp_samples, width, height, args.channels)

        mean, std = fit_normalizer(x_raw[train_idx])
        x = apply_normalizer(x_raw, mean, std)

        labels: list[str] = spec["labels"]  # type: ignore[assignment]

        save_json(model_dir / "normalization.json", {
            "format": "bmp",
            "color_mode": "RGB" if args.channels == 3 else "grayscale",
            "source_resolution": list(SOURCE_RESOLUTION),
            "image_width": width,
            "image_height": height,
            "channels": args.channels,
            "input_shape": [1, height, width, args.channels],
            "mean": mean.astype(float).tolist(),
            "std": std.astype(float).tolist(),
            "softmax_temperature": float(args.temperature),
            "preprocessing_note": (
                "Camera frames are 640x480. Resize to (image_width, image_height) "
                "using bilinear interpolation. Then feed raw 0..255 pixels normalized "
                "as (pixel - mean[channel]) / std[channel]. For int8 TFLite, normalize "
                "first, then quantize using input tensor scale/zero_point."
            ),
        })

        model_rows, num_classes = train_and_export(
            spec,
            x,
            gestures,
            conditions,
            split,
            args,
            model_dir,
            width,
            height,
        )

        rows.extend(model_rows)

        preprocessing_configs[name] = {
            "width": width,
            "height": height,
            "channels": args.channels,
            "num_classes": num_classes,
            "mean": mean,
            "std": std,
        }

    save_preprocessing_cpp_header(artifacts_dir, preprocessing_configs)

    metrics_path = artifacts_dir / "model_metrics.csv"

    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Saved metrics: {metrics_path}")
    print(f"Models + normalization.json under: {artifacts_dir}/<model>/")
    print(f"C++ preprocessing config: {artifacts_dir}/preprocessing_config.h")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
