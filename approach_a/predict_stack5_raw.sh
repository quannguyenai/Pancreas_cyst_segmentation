#!/usr/bin/env bash
# approach_a/predict_stack5_raw.sh — Stack-as-channels 2.5D inference on
# raw single-channel MRI volumes.
#
# The Dataset011 model expects 5 input channels (5 z-shifted copies per case).
# This script hides that: it takes a folder of raw single-channel MRI NIfTIs,
# materialises the 5-channel form in a temp dir, runs nnUNetv2_predict, and
# cleans up the temp dir on exit.
#
# Usage:
#   bash approach_a/predict_stack5_raw.sh <INPUT_DIR> <OUTPUT_DIR> [FOLD] [CONFIG]
#
#   INPUT_DIR   Folder of single-channel .nii.gz MRI volumes. Filenames may be
#               <stem>.nii.gz or <stem>_0000.nii.gz.
#   OUTPUT_DIR  Where to write predicted segmentations.
#   FOLD        Fold(s), space-separated (default: "0").
#   CONFIG      nnUNet configuration (default: 2d).
#
# Requires: the model to be trained (train_stack5.sh) and Dataset011 to exist
# in nnUNet_raw (so nnUNetv2_predict can find its plans/checkpoints).

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: bash approach_a/predict_stack5_raw.sh <INPUT_DIR> <OUTPUT_DIR> [FOLD] [CONFIG]" >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/set_nnunet_env.sh"

INPUT_DIR="$(cd "$1" && pwd)"
OUTPUT_DIR="$2"
FOLDS="${3:-0}"
CONFIG="${4:-2d}"
DATASET_ID=11
WINDOW=5

if [[ -x "${REPO_ROOT}/.venv/bin/python3" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python3"
else
    PYTHON_BIN="$(command -v python3)"
fi

# Temp dir for the 5-channel expansion — cleaned on exit.
STACK_DIR="$(mktemp -d --suffix=_stack5)"
cleanup() { rm -rf "${STACK_DIR}"; }
trap cleanup EXIT

mkdir -p "${OUTPUT_DIR}"

echo "========================================================"
echo "Approach A (2.5D) — raw-MRI inference"
echo "  Input     : ${INPUT_DIR}"
echo "  Output    : ${OUTPUT_DIR}"
echo "  Stack dir : ${STACK_DIR}  (temp)"
echo "  Dataset   : ${DATASET_ID}  Config: ${CONFIG}  Folds: ${FOLDS}"
echo "========================================================"

echo "[1/2] Materialising ${WINDOW}-channel stack ..."
"${PYTHON_BIN}" "${REPO_ROOT}/approach_a/shift_for_predict.py" \
    --input-dir  "${INPUT_DIR}" \
    --output-dir "${STACK_DIR}" \
    --window     "${WINDOW}"

echo ""
echo "[2/2] Running nnUNetv2_predict ..."
# shellcheck disable=SC2086
nnUNetv2_predict \
    -i "${STACK_DIR}" \
    -o "${OUTPUT_DIR}" \
    -d "${DATASET_ID}" \
    -c "${CONFIG}" \
    -f ${FOLDS} \
    --save_probabilities

echo ""
echo "Predictions saved to: ${OUTPUT_DIR}"
