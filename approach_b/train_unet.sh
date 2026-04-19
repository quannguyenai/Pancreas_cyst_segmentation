#!/usr/bin/env bash
# approach_b/train_unet.sh — Train a simple 3D U-Net on cropped pancreas ROIs.
#
# Usage:
#   bash approach_b/train_unet.sh [GPU_ID] [EXP_NAME]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${1:-0}"
EXP_NAME="${2:-approach_b_unet3d}"

if [ -x "${REPO_ROOT}/.venv/bin/python3" ]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python3"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    PYTHON_BIN="$(command -v python)"
fi

eval "$("${PYTHON_BIN}" - <<PY
import shlex, sys
sys.path.insert(0, "${REPO_ROOT}")
from configs import load_config
cfg = load_config("${REPO_ROOT}/configs/paths.yaml")
vals = {
    "TRAIN_TXT": cfg["approach_b"]["cropped_train_txt"],
    "VAL_TXT": cfg["approach_b"]["cropped_val_txt"],
}
for k, v in vals.items():
    print(f"{k}={shlex.quote(str(v))}")
PY
)"

"${PYTHON_BIN}" "${REPO_ROOT}/approach_b/prepare_cropped_dataset.py" \
    --config "${REPO_ROOT}/configs/paths.yaml" \
    --skip-nnunet

"${PYTHON_BIN}" "${REPO_ROOT}/comparison/train.py" \
    --config "${REPO_ROOT}/configs/paths.yaml" \
    --mode 3d \
    --model unet_3d \
    --gpu "${GPU}" \
    --exp "${EXP_NAME}" \
    --train-txt "${TRAIN_TXT}" \
    --val-txt "${VAL_TXT}"
