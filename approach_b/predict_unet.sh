#!/usr/bin/env bash
# approach_b/predict_unet.sh — Inference for a cropped 3D U-Net cyst model.
#
# Usage:
#   bash approach_b/predict_unet.sh [GPU_ID] [EXP_NAME]

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
    "TEST_TXT": cfg["approach_b"]["cropped_test_txt"],
    "CROPPED_PREDS": cfg["approach_b"]["unet_predictions_cropped"],
    "FULL_PREDS": cfg["approach_b"]["unet_predictions_full"],
    "TEST_STATS": cfg["approach_b"]["crop_stats_test_json"],
    "PANCREAS_PREDS": cfg["approach_b"]["pancreas_preds"],
}
for k, v in vals.items():
    print(f"{k}={shlex.quote(str(v))}")
PY
)"

mkdir -p "${CROPPED_PREDS}" "${FULL_PREDS}"

if [ ! -d "${PANCREAS_PREDS}" ] || [ -z "$(ls -A "${PANCREAS_PREDS}" 2>/dev/null)" ]; then
    bash "${REPO_ROOT}/approach_b/stage1_pancreas_seg.sh"
fi

"${PYTHON_BIN}" "${REPO_ROOT}/approach_b/crop_to_pancreas.py" \
    --config "${REPO_ROOT}/configs/paths.yaml" \
    --split test \
    --stats-path "${TEST_STATS}"

"${PYTHON_BIN}" "${REPO_ROOT}/approach_b/prepare_cropped_dataset.py" \
    --config "${REPO_ROOT}/configs/paths.yaml" \
    --skip-nnunet

"${PYTHON_BIN}" "${REPO_ROOT}/comparison/test.py" \
    --config "${REPO_ROOT}/configs/paths.yaml" \
    --mode 3d \
    --model unet_3d \
    --gpu "${GPU}" \
    --checkpoint "${REPO_ROOT}/comparison/checkpoints/${EXP_NAME}/best_model.pth" \
    --test-txt "${TEST_TXT}" \
    --output "${CROPPED_PREDS}"

"${PYTHON_BIN}" "${REPO_ROOT}/approach_b/paste_back.py" \
    --config "${REPO_ROOT}/configs/paths.yaml" \
    --crop-stats "${TEST_STATS}" \
    --cropped-preds "${CROPPED_PREDS}" \
    --output "${FULL_PREDS}"
