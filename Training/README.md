# Training — Rock-Paper-Scissors models

`pretraining.py` trains **two image models** and exports each as float32 + int8
TFLite, following the course's artifact conventions:

| model | type | classes |
|---|---|---|
| `gesture` | 3-class softmax | Rock / Paper / Scissors (bracelet-agnostic) |
| `bracelet` | binary | nothing / bracelet (gesture-agnostic, runs at higher resolution) |

The game rule lives in the **C++ program**, not the model: `system_move = bracelet ?
lose_to(gesture) : beat(gesture)`. Keeping policy out of the weights means a rule
change is a one-line edit, not a retrain, and perception errors stay debuggable.

Source images are converted to **BMP** first, so training and the Pi share one decode
path. The two models share one stratified split but each has its **own resolution,
normalizer, and artifact folder** (the bracelet is a small cue, so it gets more pixels).

## Dev container (recommended — plug-and-play)

The repo ships a `.devcontainer/` modelled on the course's. Open the repo in VS Code and
**"Reopen in Container"** — it builds an Ubuntu 22.04 image with **Python 3.10 +
TensorFlow 2.16.1** (matching the course / Pi toolchain, so the host OS and its Python
version don't matter) and installs the deps automatically. Then:

```bash
make train          # train both models, export TFLite into Training/artifacts/
make train-quick    # 3-epoch smoke test
```
or run the **`train-model`** VS Code task. The gesture dataset is expected in the sibling
`EAI4IL-project-data` repo; override with `make train DATA_DIR=/path/to/dataset` and pass
extra flags with `ARGS="--epochs 150"`. (Make sure that dataset is mounted/visible inside
the container — e.g. open the parent folder that holds both repos.)

## Manual setup (without the dev container)
- **Python 3.10** (TF 2.16 has no wheels for 3.13+)
- Packages in [`requirements.txt`](requirements.txt): TensorFlow 2.16.1, NumPy 1.26.4, Pillow

### Windows (PowerShell)
```powershell
cd rock-paper-scissors-rpi\Training
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
> If `py -3.10` reports it can't find Python 3.10, install it from python.org first.
> If `Activate.ps1` is blocked, run once:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.

### Linux / macOS / devcontainer (bash)
```bash
cd rock-paper-scissors-rpi/Training
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run training
From the `Training/` directory with the venv active:
```bash
python pretraining.py
```
The dataset path defaults to the sibling `EAI4IL-project-data/` repo (resolved
relative to this script, so the working directory doesn't matter). Common flags:

The camera produces **640×480 (4:3)** frames; the script center-crops to the model
aspect (a no-op for a 4:3 frame) and downscales, so the model input stays tiny.

| flag | default | meaning |
|---|---|---|
| `--data-dir` | sibling `EAI4IL-project-data` | dataset root (`<person>/<Gesture>_<condition>/*`) |
| `--gesture-width` / `--gesture-height` | `64` / `48` | gesture model input (4:3) |
| `--bracelet-width` / `--bracelet-height` | `128` / `96` | bracelet model input (4:3, larger — small cue) |
| `--channels` | `3` | `3` = RGB, `1` = grayscale |
| `--base-filters` / `--dense-units` | `16` / `64` | model capacity (smaller = lighter on the Pi) |
| `--prune` | off | also export HW3 synapse- and neuron-pruned variants |
| `--epochs` | `60` | max epochs (early-stopping is on) |
| `--no-augment` | off | disable training augmentation |

The net mirrors HW3's design in 2D (conv blocks → global average → dense head). Pruning
is **off by default** — on a conv-dominated net it barely shrinks the file (TFLite stores
zeroed weights densely); int8 quantization is what keeps the models small. Use `--prune`
to also emit the synapse- and neuron-pruned variants.

A quick end-to-end check on the existing images:
```bash
python pretraining.py --epochs 3
```

## Outputs (`artifacts/`, git-ignored)
One folder per model, so the C++ side reads one folder per interpreter:
- `artifacts/gesture/model_f32.tflite`, `model_i8.tflite`, `label_map.json`, `normalization.json`
- `artifacts/bracelet/model_f32.tflite`, `model_i8.tflite`, `label_map.json`, `normalization.json`
- `artifacts/model_metrics.csv` — accuracy, balanced accuracy, params, size for every model + export
- `data_bmp/<model>/` — the BMP-converted dataset per resolution (git-ignored)

Each model's `normalization.json` carries its **own** `mean`/`std` and `input_shape`
(the two models use different resolutions), plus the crop→resize→z-score note the C++
preprocessing must mirror.
