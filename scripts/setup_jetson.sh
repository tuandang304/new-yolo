#!/usr/bin/env bash
# Idempotent Jetson environment for latency nodes: venv with system site-packages (system TensorRT python bindings),
# PyTorch CUDA 13.2 aarch64 wheels (run on Orin sm_87), checks the MAXN_SUPER power mode and fixes clocks. Needs passwordless sudo.
set -euo pipefail
VENV="$HOME/venvs/bench"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv --system-site-packages "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -c "import torch, torchvision" 2>/dev/null || \
  python -m pip install -q --disable-pip-version-check --progress-bar off "torch==2.12.0+cu132" "torchvision==0.27.0+cu132" --index-url https://download.pytorch.org/whl/cu132
python -m pip install -q --disable-pip-version-check --progress-bar off -r requirements-jetson.txt
python -c "import torch, tensorrt; x = torch.randn(256, 256, device='cuda', dtype=torch.half); (x @ x).sum().item(); print('[env] jetson torch', torch.__version__, 'cuda', torch.version.cuda, 'cap', torch.cuda.get_device_capability(0), 'trt', tensorrt.__version__)"
# The power mode (MAXN_SUPER) is set once by hand: switching it may ask for a reboot. Fail if it is not active.
sudo -n nvpmodel -q | grep -q MAXN_SUPER || { echo "[fail] stage=setup error=power mode is not MAXN_SUPER: $(sudo -n nvpmodel -q | head -1)"; exit 1; }
sudo -n jetson_clocks
echo "[env] jetson $(tr -d '\0' < /proc/device-tree/model) | $(head -1 /etc/nv_tegra_release) | $(sudo -n nvpmodel -q | head -1) | jetson_clocks on"
