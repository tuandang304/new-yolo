#!/usr/bin/env bash
# Fixed run contract for every node: install pinned deps, then run the node's configs/node.yaml.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d /opt/conda/bin ] && export PATH="/opt/conda/bin:$PATH"
PY="$(command -v python || command -v python3)"
if ! "$PY" -m pip --version >/dev/null 2>&1; then
  sudo -n apt-get update -qq && sudo -n apt-get install -y -qq libxcb1 libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 || true
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  uv venv --python 3.12 --seed .venv
  export PATH="$PWD/.venv/bin:$PATH"
  PY="$PWD/.venv/bin/python"
  uv pip install -q torch torchvision -r requirements-run.txt
fi
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg YOLO_OFFLINE=0 GITHUB_ACTIONS=1
"$PY" -m pip install -q --disable-pip-version-check -r requirements-run.txt
export ORX_OUTPUT_DIR="${ORX_OUTPUT_DIR:-$PWD/outputs}"
mkdir -p "$ORX_OUTPUT_DIR"
"$PY" -m pip freeze > "$ORX_OUTPUT_DIR/pip_freeze.txt" || true
nvidia-smi || true
"$PY" -m bench.main --config configs/node.yaml
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg YOLO_OFFLINE=0
export GITHUB_ACTIONS=1  # compute logging rules: no per-batch progress bars (60 s refresh); do not set GITHUB_WORKFLOW/RUNNER_OS
python -m pip install -q --disable-pip-version-check -r requirements-run.txt
export ORX_OUTPUT_DIR="${ORX_OUTPUT_DIR:-$PWD/outputs}"
mkdir -p "$ORX_OUTPUT_DIR"
python -m pip freeze > "$ORX_OUTPUT_DIR/pip_freeze.txt" || true
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
python -m bench.main --config configs/node.yaml
