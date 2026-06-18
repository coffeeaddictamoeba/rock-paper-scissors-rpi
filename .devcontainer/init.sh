#!/usr/bin/env bash
set -euo pipefail

# Runs once when the dev container is created. The Dockerfile already put /opt/venv
# (Python 3.10) on PATH, so we just install the training deps into it and make the
# output dir. After this, training is plug-and-play: `make train`.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r Training/requirements.txt

mkdir -p Training/artifacts

echo "Dev container ready. Train with:  make train   (or: python Training/pretraining.py)"
