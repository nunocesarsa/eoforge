#!/usr/bin/env bash
# One-step install of EOForge with every model in the zoo, for WSL/Linux with an NVIDIA GPU.
#
#   bash install.sh                         # venv in ./.venv, PyTorch for GTX 10xx and newer (CUDA 12.6)
#   bash install.sh /mnt/d/venvs/eoforge    # another venv location (e.g. on a bigger drive under WSL)
#   TORCH_CUDA=cu128 TORCH_VERSION=2.14.0 TORCHVISION_VERSION=0.29.0 bash install.sh   # newer GPUs
#   SKIP_DOWNLOAD=1 bash install.sh         # do not pre-download model weights (several GB)
set -euo pipefail
cd "$(dirname "$0")"

VENV="${1:-$(pwd)/.venv}"
TORCH_CUDA="${TORCH_CUDA:-cu126}"              # cu126 = last build supporting Pascal (GTX 10xx)
TORCH_VERSION="${TORCH_VERSION:-2.14.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.29.0}"
CONSTRAINTS="$(mktemp)"
printf "torch==%s\ntorchvision==%s\n" "$TORCH_VERSION" "$TORCHVISION_VERSION" > "$CONSTRAINTS"

# PyTorch wheels are large: set TMPDIR to a folder on a drive with space if /tmp is small,
# e.g.  TMPDIR=/mnt/d/wsl_tmp bash install.sh   under WSL
[ -n "${TMPDIR:-}" ] && mkdir -p "$TMPDIR"

echo "== Python environment: $VENV"
[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --no-cache-dir --upgrade pip setuptools wheel

echo "== PyTorch $TORCH_VERSION ($TORCH_CUDA)"
pip install --no-cache-dir "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
    --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"

echo "== EOForge and all model libraries"
pip install --no-cache-dir -e . -c "$CONSTRAINTS"

echo "== SAM 2 (built against the installed PyTorch, without the optional CUDA extension)"
SAM2_BUILD_CUDA=0 pip install --no-cache-dir --no-build-isolation sam2 hydra-core iopath -c "$CONSTRAINTS"

python - <<'PY'
import torch
print("PyTorch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none found")
PY

if [ "${SKIP_DOWNLOAD:-0}" != "1" ]; then
  echo "== Downloading model weights (one time, several GB)"
  eoforge zoo download || echo "Some downloads failed; rerun 'eoforge zoo download' later."
fi
eoforge zoo status
rm -f "$CONSTRAINTS"
echo
echo "Done. Start the app with:"
echo "  source $VENV/bin/activate && eoforge app --folder $(pwd)"
