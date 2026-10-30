#!/usr/bin/env bash
# Fixed run contract for every node: install pinned deps, then run the node's configs/node.yaml.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg YOLO_OFFLINE=0
python -m pip install -q --disable-pip-version-check -r requirements-run.txt
export ORX_OUTPUT_DIR="${ORX_OUTPUT_DIR:-$PWD/outputs}"
mkdir -p "$ORX_OUTPUT_DIR"
python -m pip freeze > "$ORX_OUTPUT_DIR/pip_freeze.txt" || true
nvidia-smi || true
python -m bench.main --config configs/node.yaml
