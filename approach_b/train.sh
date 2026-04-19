#!/usr/bin/env bash
# approach_b/train.sh — Stage 2: Train nnUNet on cropped pancreas-ROI volumes
# (Dataset010_CroppedCyst).
#
# Prerequisites:
#   1. Stage 1 complete: bash approach_b/stage1_pancreas_seg.sh
#   2. Cropping complete:
#        python approach_b/crop_to_pancreas.py --config configs/paths.yaml
#   3. Register cropped volumes as nnUNet dataset:
#        python approach_b/prepare_cropped_dataset.py --config configs/paths.yaml
#   4. Preprocess:
#        nnUNetv2_plan_and_preprocess -d 10 --verify_dataset_integrity -np 8
#
# Usage:
#   bash approach_b/train.sh [FOLD]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/set_nnunet_env.sh"

FOLD="${1:-0}"
CONFIG="3d_fullres"
if [ -x "${REPO_ROOT}/.venv/bin/python3" ]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python3"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    PYTHON_BIN="$(command -v python)"
fi

DATASET_ID="$("${PYTHON_BIN}" - <<PY
import sys
sys.path.insert(0, "${REPO_ROOT}")
from configs import load_config
cfg = load_config("${REPO_ROOT}/configs/paths.yaml")
print(cfg["approach_b"]["nnunet_dataset_id"])
PY
)"
DATASET_SUFFIX="$("${PYTHON_BIN}" - <<PY
import sys
sys.path.insert(0, "${REPO_ROOT}")
from configs import load_config
cfg = load_config("${REPO_ROOT}/configs/paths.yaml")
print(cfg["approach_b"]["nnunet_dataset_name"])
PY
)"

DATASET_NAME="Dataset$(printf '%03d' "${DATASET_ID}")_${DATASET_SUFFIX}"

if [ -x "${REPO_ROOT}/.venv/bin/nnUNetv2_train" ]; then
    NNUNET_TRAIN="${REPO_ROOT}/.venv/bin/nnUNetv2_train"
else
    NNUNET_TRAIN="nnUNetv2_train"
fi

echo "========================================================"
echo "Approach B — Stage 2: Cyst segmentation on cropped volumes"
echo "  Dataset  : ${DATASET_NAME}"
echo "  Config   : ${CONFIG}"
echo "  Fold     : ${FOLD}"
echo "========================================================"

# Verify dataset exists
if [ ! -d "${nnUNet_raw}/${DATASET_NAME}" ]; then
    echo "[ERROR] Dataset not found: ${nnUNet_raw}/${DATASET_NAME}"
    echo ""
    echo "  Run the full setup sequence described in this script's prerequisites."
    exit 1
fi

"${NNUNET_TRAIN}" "${DATASET_ID}" "${CONFIG}" "${FOLD}" --npz

echo "Training complete."
