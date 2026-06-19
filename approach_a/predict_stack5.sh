#!/usr/bin/env bash
# approach_a/predict_stack5.sh — Run nnU-Net inference on the 2.5D test set
# (Dataset011_PancreasCyst25D, config `2d`).
#
# Usage:
#   bash approach_a/predict_stack5.sh [FOLD] [CONFIG]
#
#   FOLD    Fold(s) to use for ensemble, space-separated (default: "0").
#   CONFIG  nnUNet configuration (default: 2d).
#
# Output: approach_a/predictions/2d_stack5/

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/set_nnunet_env.sh"

FOLDS="${1:-0}"
CONFIG="${2:-2d}"
DATASET_ID=11
DATASET_NAME="PancreasCyst25D"

INPUT_DIR="${nnUNet_raw}/Dataset$(printf '%03d' ${DATASET_ID})_${DATASET_NAME}/imagesTs"
OUTPUT_DIR="${REPO_ROOT}/approach_a/predictions/2d_stack5"

if [ ! -d "${INPUT_DIR}" ]; then
    echo "[ERROR] Test images not found at: ${INPUT_DIR}"
    echo "        Run: python approach_a/prepare_stack5_dataset.py --config configs/paths.yaml"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "========================================================"
echo "Approach A (2.5D) — inference"
echo "  Dataset  : Dataset$(printf '%03d' ${DATASET_ID})_${DATASET_NAME}"
echo "  Config   : ${CONFIG}"
echo "  Fold(s)  : ${FOLDS}"
echo "  Input    : ${INPUT_DIR}"
echo "  Output   : ${OUTPUT_DIR}"
echo "========================================================"

# shellcheck disable=SC2086
nnUNetv2_predict \
    -i "${INPUT_DIR}" \
    -o "${OUTPUT_DIR}" \
    -d "${DATASET_ID}" \
    -c "${CONFIG}" \
    -f ${FOLDS} \
    --save_probabilities

echo ""
echo "Predictions saved to: ${OUTPUT_DIR}"
