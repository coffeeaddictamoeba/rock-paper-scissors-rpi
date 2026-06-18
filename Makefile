# Convenience targets for the dev container. Training runs in the container's
# /opt/venv (Python 3.10 + TensorFlow 2.16.1), matching the course / Pi toolchain.
#
# The gesture dataset lives in the sibling EAI4IL-project-data repo; override its
# location with DATA_DIR=... and pass extra flags with ARGS="...".

PY ?= python3
TRAIN := Training/pretraining.py
DATA_DIR ?= ../EAI4IL-project-data

.PHONY: train train-quick deps

# Train the gesture + bracelet models and export TFLite into Training/artifacts/.
train:
	$(PY) $(TRAIN) --data-dir $(DATA_DIR) $(ARGS)

# Fast end-to-end smoke (3 epochs) to confirm the pipeline runs.
train-quick:
	$(PY) $(TRAIN) --data-dir $(DATA_DIR) --epochs 3 $(ARGS)

# (Re)install the Python training dependencies.
deps:
	$(PY) -m pip install -r Training/requirements.txt
