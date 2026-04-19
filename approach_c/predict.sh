#!/usr/bin/env bash
# approach_c/predict.sh — Inference with nnTransUNetTrainerV2_Pretrained.
#
# Uses nnUNet v1 predict (same env as approach_d).
#
# Usage (from repo root):
#   bash approach_c/predict.sh [FOLD] [CKPT_NAME]
#
# Arguments:
#   FOLD       Fold to use for prediction (default: 0)
#   CKPT_NAME  Checkpoint filename (default: model_best)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── Activate approach_d venv + nnUNet v1 env vars ─────────────────────────────
# shellcheck disable=SC1091
source approach_d/set_env.sh

TASK_ID=1
TASK_NAME="PancreasCyst"
TRAINER="nnTransUNetTrainerV2_Pretrained"
CONFIG="3d_fullres"
FOLD="${1:-0}"
CKPT="${2:-model_best}"

# ── Install custom trainer into nnUNet package ────────────────────────────────
NNUNET_TRAINER_DIR="$(python3 -c "import nnunet, os; print(os.path.join(os.path.dirname(nnunet.__file__), 'training', 'network_training'))")"
cp approach_c/nnTransUNetTrainerV2_Pretrained.py "${NNUNET_TRAINER_DIR}/"

INPUT_DIR="${nnUNet_raw_data_base}/nnUNet_raw_data/Task$(printf '%03d' ${TASK_ID})_${TASK_NAME}/imagesTs"
OUTPUT_DIR="${REPO_ROOT}/approach_c/predictions"

mkdir -p "${OUTPUT_DIR}"

echo "========================================================"
echo "  Approach C — nnTransUNetTrainerV2_Pretrained predict"
echo "  Task   : Task$(printf '%03d' ${TASK_ID})_${TASK_NAME}"
echo "  Fold   : ${FOLD}"
echo "  Ckpt   : ${CKPT}"
echo "  Input  : ${INPUT_DIR}"
echo "  Output : ${OUTPUT_DIR}"
echo "========================================================"

nnUNet_predict \
    -tr  "${TRAINER}" \
    -i   "${INPUT_DIR}" \
    -o   "${OUTPUT_DIR}" \
    -t   "${TASK_ID}" \
    -m   "${CONFIG}" \
    --folds "${FOLD}" \
    -chk "${CKPT}"
