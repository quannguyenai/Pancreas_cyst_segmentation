#!/usr/bin/env bash
# approach_a/train_mixed.sh — Approach A1: Train nnUNet 3d_fullres on the full
# mixed dataset (Dataset001_PancreasCyst, all 247 training cases).
#
# Usage:
#   bash approach_a/train_mixed.sh [FOLD] [CONFIG]
#
# Arguments:
#   FOLD    Fold index to train (default: 0).
#           To train all 5 folds: for f in 0 1 2 3 4; do bash ... $f; done
#   CONFIG  nnUNet configuration name (default: 3d_fullres).
#
# Prerequisites:
#   1. Data prepared: python data/prepare_dataset.py --config configs/paths.yaml \
#          --update-txts --build-nnunet
#   2. nnUNet preprocessed: run approach_a/preprocess.sh (or manually:
#          nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity)
#   3. nnUNet installed in active Python environment.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/set_nnunet_env.sh"

FOLD="${1:-0}"
CONFIG="${2:-3d_fullres}"
DATASET_ID=1

echo "========================================================"
echo "Approach A1 — Full mixed dataset training"
echo "  Dataset  : Dataset$(printf '%03d' ${DATASET_ID})_PancreasCyst"
echo "  Config   : ${CONFIG}"
echo "  Fold     : ${FOLD}"
echo "  Results  : ${nnUNet_results}"
echo "========================================================"

nnUNetv2_train "${DATASET_ID}" "${CONFIG}" "${FOLD}" --npz

echo "Training complete. Checkpoint saved to:"
echo "  ${nnUNet_results}/Dataset$(printf '%03d' ${DATASET_ID})_PancreasCyst/nnUNetTrainer__nnUNetPlans__${CONFIG}/fold_${FOLD}/"
