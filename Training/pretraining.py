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


# Two models, both the HW3 sensor net reshaped to 2D (conv blocks -> global average
# -> dense head):
#   gesture  : 3-class softmax {Rock, Paper, Scissors}, bracelet-agnostic
#   bracelet : binary {nothing, bracelet}, run at a higher resolution because the
#              bracelet is a small cue that gets washed out at the gesture size
# The game rule (bracelet -> system loses, else wins) lives in the C++ program, not
# in the weights, so the models stay perceptual and a rule change needs no retrain.

GESTURES = ["Rock", "Paper", "Scissors"]
CONDITIONS = ["bracelet", "nothing"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# The Pi camera frames are 640x480 (4:3). We center-crop to the model's aspect
# (a no-op for an already-4:3 frame) and downscale, so the model stays Pi-sized.
# The C++ side mirrors this: resize 640x480 -> (img_width, img_height).
SOURCE_RESOLUTION = (640, 480)

BASE_FILTERS = 16
DENSE_UNITS = 64
DROPOUT = 0.3  # higher than HW3 (0.125): the tiny image set overfits fast
L2 = 1e-4
LEARNING_RATE = 1e-3
NOISE_STD = 0.03
REPRESENTATIVE_SAMPLES = 200
SOFTMAX_TEMPERATURE = 0.5

# One entry per model. label_of(gesture, condition) returns the label string; the
# class id is its index in labels. The gesture model ignores the bracelet condition
# (both wrist conditions are training variation) and the bracelet model ignores the
# gesture. Default resolution is per model; override with the CLI flags below.
MODELS: list[dict[str, object]] = [
    {
        "name": "gesture",
        "labels": GESTURES,
        "label_of": lambda g, c: g,
        "width": 64,
        "height": 48,
    },
    {
        "name": "bracelet",
        "labels": ["nothing", "bracelet"],
        "label_of": lambda g, c: c,
        "width": 128,
        "height": 96,
    },
]

# Dataset lives in the sibling EAI4IL-project-data repo. Resolve it relative to this
# file so the script works no matter the current working directory.
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "EAI4IL-project-data"


def parse_args():
    parser = argparse.ArgumentParser(description="Train the RPS gesture + bracelet image models and export TFLite.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Dataset root, e.g. <person>/Rock_bracelet/*.jpg")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Output directory (one subfolder per model)")
    parser.add_argument("--bmp-dir", default="data_bmp", help="Where source images are materialized as BMP (one subfolder per model)")
    parser.add_argument("--channels", type=int, default=3, choices=(1, 3), help="3 = RGB, 1 = grayscale")

    # per-model input size (camera is 640x480, 4:3). bracelet runs larger on purpose.
    parser.add_argument("--gesture-width", type=int, default=64)
    parser.add_argument("--gesture-height", type=int, default=48)
    parser.add_argument("--bracelet-width", type=int, default=128)
    parser.add_argument("--bracelet-height", type=int, default=96)

    parser.add_argument("--base-filters", type=int, default=BASE_FILTERS, help="Width of the first conv block")
    parser.add_argument("--dense-units", type=int, default=DENSE_UNITS, help="Units in the hidden dense layer")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--no-augment", action="store_true", help="Disable training augmentation")

    parser.add_argument(
        "--temperature",
        type=float,
        default=SOFTMAX_TEMPERATURE,
        help=(
            "Softmax sharpening temperature for exported TFLite models. "
            "1.0 keeps original confidence. Lower values increase confidence."
        ),
    )

    # HW3-style pruning. Off by default: it barely shrinks a conv-dominated net (synapse
    # pruning zeros weights but TFLite still stores them densely; neuron pruning only
    # trims the small hidden dense). Turn on to export the pruned variants too.
    parser.add_argument("--prune", action="store_true", help="Also export synapse- and neuron-pruned variants")
    parser.add_argument("--synapse-prune-ratio", type=float, default=0.50, help="Fraction of smallest kernel weights to zero")
    parser.add_argument("--neuron-prune-ratio", type=float, default=0.25, help="Fraction of hidden dense neurons to remove")

    return parser.parse_args()


# Dataset loading

# Expected layout (labels come from folder names):
# data/
#   <person>/
#     Rock_bracelet/
#       img1.jpg
#     Rock_nothing/
#     Paper_bracelet/
#     ...
# Persons are pooled together; only the gesture and the bracelet condition matter.


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


# center-crop an image to the target width:height aspect (no-op when it already matches).
def crop_to_aspect(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    w, h = img.size
    target_ratio = target_w / target_h
    ratio = w / h

    if ratio > target_ratio:
        # too wide: trim the sides
        new_w = int(round(h * target_ratio))
        left = (w - new_w) // 2
        return img.crop((left, 0, left + new_w, h))

    if ratio < target_ratio:
        # too tall: trim top and bottom
        new_h = int(round(w / target_ratio))
        top = (h - new_h) // 2
        return img.crop((0, top, w, top + new_h))

    return img


# Convert every source image to BMP so training decodes exactly what the Pi decodes.
def convert_to_bmp(samples: list[tuple[Path, str, str, str]], bmp_dir: Path, img_width: int, img_height: int, channels: int
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


# Load the BMPs into one [N, H, W, C] array of raw pixels (0..255). Normalization
# happens later so we can fit it on the training split only.
def load_images(bmp_samples: list[tuple[Path, str, str, str]], img_width: int, img_height: int, channels: int) -> np.ndarray:
    mode = "RGB" if channels == 3 else "L"

    x = np.zeros((len(bmp_samples), img_height, img_width, channels), dtype=np.float32)

    for i, (path, _person, _gesture, _condition) in enumerate(bmp_samples):
        with Image.open(path) as img:
            cropped = crop_to_aspect(img.convert(mode), img_width, img_height)
            arr = np.asarray(cropped.resize((img_width, img_height), Image.BILINEAR), dtype=np.float32)

        if channels == 1:
            arr = arr[..., None]

        x[i] = arr

    return x


# Per-stratum split so all six gesture_condition groups stay balanced across train/val/test.
def get_train_and_test_data(strata: np.ndarray, val_ratio: float, test_ratio: float, seed: int
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


# Fit per-channel mean/std on the training pixels only (raw 0..255 scale).
def fit_normalizer(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = x_train.reshape(-1, x_train.shape[-1])

    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)

    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    return mean, std


def apply_normalizer(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean.reshape(1, 1, 1, -1)) / std.reshape(1, 1, 1, -1)).astype(np.float32)


def compute_class_weights(y: np.ndarray, num_classes: int) -> dict[int, float]:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    total = float(np.sum(counts))

    return {
        cls: total / (num_classes * float(count))
        for cls, count in enumerate(counts)
        if count > 0
    }


# stronger augmentation than HW3: the image set is tiny so a from-scratch net
# memorizes it within a few epochs. reflect-fill keeps rotated/shifted borders
# from turning into a flat color.
_AUGMENTER = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.08, fill_mode="reflect"),
    tf.keras.layers.RandomZoom(0.1, fill_mode="reflect"),
    tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode="reflect"),
])


def augment_batch(x: tf.Tensor, y: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    x = _AUGMENTER(x, training=True)

    x = x + tf.random.normal(
        tf.shape(x),
        mean=0.0,
        stddev=NOISE_STD,
        dtype=x.dtype,
    )

    return x, y


def make_dataset(x: np.ndarray, y: np.ndarray, batch_size: int, training: bool, seed: int, augment: bool = False) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((x.astype(np.float32), y.astype(np.int64)))

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


# Model

def conv_block(x: tf.Tensor, filters: int, kernel_size: int, dropout: float, pool: bool) -> tf.Tensor:
    reg = tf.keras.regularizers.l2(L2) if L2 > 0.0 else None

    for k in (kernel_size, 3):
        x = tf.keras.layers.Conv2D(
            filters,
            k,
            padding="same",
            use_bias=False,
            kernel_regularizer=reg,
        )(x)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

    if pool:
        x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)

    if dropout > 0.0:
        x = tf.keras.layers.Dropout(dropout)(x)

    return x


def make_model(img_width: int, img_height: int, channels: int, num_classes: int, name: str, base_filters: int, dense_units: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(img_height, img_width, channels), name="image")

    # pool in every block so we downsample quickly and keep the Pi cost low.
    x = conv_block(inputs, base_filters,     kernel_size=7, dropout=DROPOUT * 0.5, pool=True)
    x = conv_block(x,      base_filters * 2, kernel_size=5, dropout=DROPOUT,       pool=True)
    x = conv_block(x,      base_filters * 3, kernel_size=3, dropout=DROPOUT,       pool=True)

    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average")(x)

    x = tf.keras.layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(L2) if L2 > 0.0 else None,
        name="hidden",
    )(x)

    if DROPOUT > 0.0:
        x = tf.keras.layers.Dropout(DROPOUT)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    return model


def make_confident_deployment_model(trained_softmax_model: tf.keras.Model, temperature: float) -> tf.keras.Model:
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")

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
        # prune only real trainable kernels, not BatchNorm gamma/beta variables.
        if hasattr(layer, "kernel"):
            kernels.append(np.abs(layer.kernel.numpy()).reshape(-1))

    if not kernels:
        raise ValueError("No prunable kernels found.")

    all_kernel_values = np.concatenate(kernels)

    # calculate the percentile to fulfill our set pruning ratio.
    threshold = np.percentile(all_kernel_values, prune_ratio * 100.0)

    # we set abs(weights) < threshold to 0.0.
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


# Helper function to define how many neurons in hidden layer to keep.
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

    # count old and calculate new hidden neurons after pruning.
    old_units = hidden_kernel.shape[1]
    new_units = keep_count(old_units, prune_ratio)

    # score each hidden neuron by incoming hidden weights plus outgoing class weights.
    incoming_scores = np.sum(np.abs(hidden_kernel), axis=0)
    outgoing_scores = np.sum(np.abs(output_kernel), axis=1)
    scores = incoming_scores + outgoing_scores

    # sort neurons by increasing importance and keep the strongest neurons.
    keep_indices = np.sort(np.argsort(scores)[-new_units:])

    # rebuild the model with reduced hidden size.
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


# TFLite export and evaluation

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


def export_tflite_models(model: tf.keras.Model, model_dir: Path, x_train: np.ndarray, file_stem: str) -> dict[str, Path]:
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


def evaluate_tflite(path: Path, x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
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

    preds_arr = np.asarray(preds)

    accuracy = float((preds_arr == y).mean()) if len(x) else float("nan")

    # balanced accuracy = mean per-class recall, so an imbalanced test set can't be
    # gamed by always predicting the majority class.
    recalls = [float((preds_arr[y == cls] == cls).mean()) for cls in np.unique(y)]
    balanced = float(np.mean(recalls)) if recalls else float("nan")

    avg_confidence = float(np.mean(confidences)) if confidences else float("nan")

    return accuracy, balanced, avg_confidence


def count_parameters(model: tf.keras.Model) -> int:
    return int(np.sum([np.prod(w.shape) for w in model.weights]))


def file_kb(path: Path) -> float:
    return path.stat().st_size / 1024.0


def save_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# Train one model, then export the original (and the pruned variants if --prune).
def train_and_export(spec: dict, x: np.ndarray, gestures: list[str], conditions: list[str],
                     split: tuple[np.ndarray, np.ndarray, np.ndarray], args, model_dir: Path,
                     img_width: int, img_height: int) -> list[dict]:
    name = str(spec["name"])
    labels: list[str] = spec["labels"]  # type: ignore[assignment]
    label_of: Callable[[str, str], str] = spec["label_of"]  # type: ignore[assignment]
    num_classes = len(labels)

    # class ids come straight from the folder names via this model's label function.
    y = np.asarray([labels.index(label_of(g, c)) for g, c in zip(gestures, conditions)], dtype=np.int64)

    train_idx, val_idx, test_idx = split
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    print(f"\n=== {name} ({num_classes}-class, {img_width}x{img_height}) ===")
    print(f"  train class counts: {np.bincount(y_train, minlength=num_classes).tolist()}  labels={labels}")

    train_ds = make_dataset(x_train, y_train, args.batch_size, training=True, seed=args.seed, augment=not args.no_augment)
    val_ds = make_dataset(x_val, y_val, args.batch_size, False, args.seed) if len(x_val) else None
    test_ds = make_dataset(x_test, y_test, args.batch_size, False, args.seed) if len(x_test) else None

    model = make_model(img_width, img_height, args.channels, num_classes, name, args.base_filters, args.dense_units)

    monitor = "val_accuracy" if val_ds is not None else "accuracy"

    # val is noisy on so few images, so be patient before stopping and keep the best epoch.
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, mode="max", factor=0.5, patience=10, min_lr=1e-5, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor=monitor, mode="max", patience=30, restore_best_weights=True, verbose=1),
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

    # the original trained net, optionally plus the two HW3 pruned variants.
    variants = [("model", model, "original")]

    if args.prune:
        variants.append(("model_synapse_pruned", prune_synapses(model, args.synapse_prune_ratio), f"synapse prune ratio={args.synapse_prune_ratio}"))
        variants.append(("model_neuron_pruned", prune_neurons(model, args.neuron_prune_ratio), f"neuron prune ratio={args.neuron_prune_ratio}"))

    rows: list[dict] = []

    for file_stem, base, note in variants:
        # this is the model used by C++ / Pi: temperature-sharpened, then quantized.
        deployment_model = make_confident_deployment_model(base, args.temperature)
        tflite_paths = export_tflite_models(deployment_model, model_dir, x_train, file_stem)

        for kind, path in tflite_paths.items():
            if len(x_test):
                acc, bal, conf = evaluate_tflite(path, x_test, y_test)
            else:
                acc, bal, conf = float("nan"), float("nan"), float("nan")

            rows.append({
                "model": name,
                "variant": f"{file_stem}_{kind}",
                "file": f"{name}/{path.name}",
                "accuracy": f"{acc:.6f}",
                "balanced_accuracy": f"{bal:.6f}",
                "parameters": count_parameters(base),
                "size_kb": f"{file_kb(path):.1f}",
                "notes": f"{note}, {kind}, temperature={args.temperature}, avg_confidence={conf:.4f}",
            })

            print(f"  exported {path.name:<26} acc={acc:.4f} bal_acc={bal:.4f} size={file_kb(path):6.1f} KB")

    return rows


def main():
    args = parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # discover once; the gesture/condition labels and the split do not depend on resolution.
    samples = discover_samples(Path(args.data_dir))
    gestures = [g for _p, _person, g, _c in samples]
    conditions = [c for _p, _person, _g, c in samples]

    # one shared stratified split over the six gesture_condition groups
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

    rows: list[dict] = []

    # each model gets its own resolution, BMP set, normalizer, and artifact folder.
    for spec in MODELS:
        name = str(spec["name"])
        width, height = sizes[name]
        model_dir = artifacts_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)

        bmp_samples = convert_to_bmp(samples, Path(args.bmp_dir) / name, width, height, args.channels)
        x_raw = load_images(bmp_samples, width, height, args.channels)

        mean, std = fit_normalizer(x_raw[train_idx])
        x = apply_normalizer(x_raw, mean, std)

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
                "Camera frames are 640x480 (4:3). Center-crop to the model aspect (no-op for a 4:3 frame), "
                "resize to (image_width, image_height) bilinear, then per-channel z-score (px - mean) / std "
                "on the raw 0..255 pixel scale. Same path for training and the Pi. For int8 TFLite, normalize "
                "first, then quantize using the input tensor scale/zero_point from the TFLite interpreter."
            ),
        })

        rows.extend(train_and_export(spec, x, gestures, conditions, split, args, model_dir, width, height))

    metrics_path = artifacts_dir / "model_metrics.csv"

    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Saved metrics: {metrics_path}")
    print(f"Models + per-model normalization.json under: {artifacts_dir}/<model>/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
