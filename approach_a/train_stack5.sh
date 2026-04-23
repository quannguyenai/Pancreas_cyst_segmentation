#!/usr/bin/env bash
# approach_a/train_stack5.sh — Approach A (2.5D): Train nnU-Net `2d` on
# Dataset011_PancreasCyst25D, where each case's CT is stored as 5 z-shifted
# copies (channels _0000.._0004). The built-in 2D U-Net sees a 5-channel
# slice input — stack-as-channels 2.5D.
#
# Usage:
#   bash approach_a/train_stack5.sh [FOLD] [CONFIG]
#
# Arguments:
#   FOLD    Fold index (default: 0). Run 0..4 for 5-fold ensemble.
#   CONFIG  nnUNet configuration (default: 2d).
#
# Prerequisites:
#   1. Dataset001 prepared (same split source):
#        python data/prepare_dataset.py --config configs/paths.yaml \
#            --update-txts --build-nnunet
#   2. Build the 2.5D dataset (one-time, ~5x disk of Dataset001 images):
#        python approach_a/prepare_stack5_dataset.py --config configs/paths.yaml

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/set_nnunet_env.sh"

FOLD="${1:-0}"
CONFIG="${2:-2d}"
DATASET_ID=11
DATASET_NAME="PancreasCyst25D"

DATASET_DIR="${nnUNet_raw}/Dataset$(printf '%03d' ${DATASET_ID})_${DATASET_NAME}"
if [[ ! -d "${DATASET_DIR}" ]]; then
    echo "[ERROR] Dataset not found: ${DATASET_DIR}"
    echo "        Run: python approach_a/prepare_stack5_dataset.py --config configs/paths.yaml"
    exit 1
fi

# Plan & preprocess if not done yet (cheap no-op on reruns)
PREPROC_MARKER="${nnUNet_preprocessed}/Dataset$(printf '%03d' ${DATASET_ID})_${DATASET_NAME}/nnUNetPlans.json"
if [[ ! -f "${PREPROC_MARKER}" ]]; then
    echo "[INFO] Running nnUNetv2_plan_and_preprocess for Dataset${DATASET_ID} ..."
    nnUNetv2_plan_and_preprocess -d "${DATASET_ID}" --verify_dataset_integrity -np 8
else
    echo "[INFO] Preprocessing already done for Dataset${DATASET_ID}, skipping."
fi

RESULTS_DIR="${nnUNet_results}/Dataset$(printf '%03d' ${DATASET_ID})_${DATASET_NAME}/nnUNetTrainer__nnUNetPlans__${CONFIG}/fold_${FOLD}"
LATEST_CKPT="${RESULTS_DIR}/checkpoint_latest.pth"

RESUME_FLAG=""
if [[ -f "${LATEST_CKPT}" ]]; then
    RESUME_FLAG="--c"
    echo "[INFO] Checkpoint found — resuming from ${LATEST_CKPT}"
fi

echo "========================================================"
echo "Approach A (2.5D) — Stack-as-channels 2D training"
echo "  Dataset  : Dataset$(printf '%03d' ${DATASET_ID})_${DATASET_NAME}"
echo "  Config   : ${CONFIG}   (5-channel 2D U-Net)"
echo "  Fold     : ${FOLD}"
echo "  Results  : ${nnUNet_results}"
echo "  Resume   : ${RESUME_FLAG:-no}"
echo "========================================================"

nnUNetv2_train "${DATASET_ID}" "${CONFIG}" "${FOLD}" --npz ${RESUME_FLAG}

echo "Training complete. Checkpoint saved to:"
echo "  ${RESULTS_DIR}/"
