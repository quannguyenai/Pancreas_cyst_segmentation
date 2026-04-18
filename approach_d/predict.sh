#!/usr/bin/env bash
# approach_d/predict.sh — Run inference with trained nnTransUNetTrainerV2.
#
# Usage (from repo root):
#   bash approach_d/predict.sh
#   bash approach_d/predict.sh --checkpoint_name model_best   # default: model_best

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source approach_d/set_env.sh

TASK_ID=1
TASK_NAME="PancreasCyst"
TRAINER="nnTransUNetTrainerV2"
CONFIG="3d_fullres"
FOLD=0
CKPT="${1:-model_best}"

INPUT_DIR="${nnUNet_raw_data_base}/nnUNet_raw_data/Task$(printf '%03d' ${TASK_ID})_${TASK_NAME}/imagesTs"
OUTPUT_DIR="${REPO_ROOT}/approach_d/predictions"
mkdir -p "${OUTPUT_DIR}"

echo "========================================================"
echo "  Approach D — nnTransUNetTrainerV2 inference"
echo "  Input    : ${INPUT_DIR}"
echo "  Output   : ${OUTPUT_DIR}"
echo "  Checkpoint: ${CKPT}"
echo "========================================================"

nnUNet_predict \
    -tr "${TRAINER}" \
    -i  "${INPUT_DIR}" \
    -o  "${OUTPUT_DIR}" \
    -t  "${TASK_ID}" \
    -m  "${CONFIG}" \
    --folds "${FOLD}" \
    --checkpoint_name "${CKPT}"

echo ""
echo "Predictions saved to: ${OUTPUT_DIR}"
