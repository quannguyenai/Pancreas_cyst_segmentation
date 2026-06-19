#!/usr/bin/env bash
# approach_a/predict.sh — Run nnUNet inference on the test set.
#
# Usage:
#   bash approach_a/predict.sh [FOLD] [CONFIG]
#
#   FOLD    Fold(s) to use for ensemble, space-separated (default: "0").
#           For 5-fold ensemble: bash approach_a/predict.sh "0 1 2 3 4"
#   CONFIG  nnUNet configuration (default: 3d_fullres).
#
# Output is written to approach_a/predictions/<CONFIG>/
# Run comparison/evaluate.py afterwards to compute Dice / HD95 / ASD.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/set_nnunet_env.sh"

FOLDS="${1:-0}"
CONFIG="${2:-3d_fullres}"
DATASET_ID=1

INPUT_DIR="${nnUNet_raw}/Dataset$(printf '%03d' ${DATASET_ID})_PancreasCyst/imagesTs"
OUTPUT_DIR="${REPO_ROOT}/approach_a/predictions/${CONFIG}"

if [ ! -d "${INPUT_DIR}" ]; then
    echo "[ERROR] Test images not found at: ${INPUT_DIR}"
    echo "        Run data/prepare_dataset.py --build-nnunet first."
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "========================================================"
echo "Approach A — nnUNet inference"
echo "  Dataset  : Dataset$(printf '%03d' ${DATASET_ID})_PancreasCyst"
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
echo "To evaluate: python comparison/evaluate.py \\"
echo "    --config configs/paths.yaml \\"
echo "    --gt-dir data/masks \\"
echo "    --pred-dirs approach_a=${OUTPUT_DIR} \\"
echo "    --output results/comparison_table.csv"
