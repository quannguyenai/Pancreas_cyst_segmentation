#!/usr/bin/env bash
# approach_b/train.sh — Stage 2: Train nnUNet on cropped pancreas-ROI volumes
# (Dataset010_CroppedCyst).
#
# Prerequisites:
#   1. Stage 1 complete: bash approach_b/stage1_pancreas_seg.sh
#   2. Cropping complete:
#        python approach_b/crop_to_pancreas.py --config configs/paths.yaml
#   3. Register cropped volumes as nnUNet dataset:
#        python data/prepare_dataset.py --config configs/paths.yaml \
#            --build-nnunet --dataset-id 10
#        (Uses approach_b/cropped/images + approach_b/cropped/masks)
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
DATASET_ID=10  # Dataset010_CroppedCyst

DATASET_NAME="Dataset$(printf '%03d' ${DATASET_ID})_CroppedCyst"

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

nnUNetv2_train "${DATASET_ID}" "${CONFIG}" "${FOLD}" --npz

echo "Training complete."
