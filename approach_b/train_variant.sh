#!/usr/bin/env bash
# approach_b/train_variant.sh — Train nnUNet for an Approach-B ROI redesign variant.
#
# Variants (see configs/paths.yaml approach_b block):
#   fixedbox   → Dataset011_FixedBoxCyst  (1-ch, fixed-box hard crop)
#   priorchan  → Dataset012_PriorChanCyst (2-ch: MRI + pancreas distance map)
#
# Prerequisites (run inside container f449bb5e955f):
#   1. Stage 1 pancreas masks exist (approach_b/stage1_pancreas_seg.sh)
#   2. Crop:    python approach_b/crop_to_pancreas.py --config configs/paths.yaml \
#                   --mode fixed-box [--emit-distance-channel]   # prior only for priorchan
#   3. Build:   python approach_b/prepare_cropped_dataset.py --variant <fixedbox|priorchan>
#   4. Preproc: nnUNetv2_plan_and_preprocess -d <ID> --verify_dataset_integrity -np 8
#   5. Splits:  python approach_b/write_splits_final.py --variant <fixedbox|priorchan>
#               (fold 0 = project train/val; do NOT use nnUNet's random 5-fold CV)
#
# Usage:
#   bash approach_b/train_variant.sh <fixedbox|priorchan> [FOLD]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/set_nnunet_env.sh"

VARIANT="${1:?usage: train_variant.sh <fixedbox|priorchan> [FOLD]}"
FOLD="${2:-0}"
CONFIG="3d_fullres"

if [ -x "${REPO_ROOT}/.venv/bin/python3" ]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python3"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    PYTHON_BIN="$(command -v python)"
fi

read -r DATASET_ID DATASET_SUFFIX <<<"$("${PYTHON_BIN}" - "${VARIANT}" <<PY
import sys
sys.path.insert(0, "${REPO_ROOT}")
from configs import load_config
cfg = load_config("${REPO_ROOT}/configs/paths.yaml")["approach_b"]
v = sys.argv[1]
print(cfg[f"{v}_dataset_id"], cfg[f"{v}_dataset_name"])
PY
)"

DATASET_NAME="Dataset$(printf '%03d' "${DATASET_ID}")_${DATASET_SUFFIX}"

if [ -x "${REPO_ROOT}/.venv/bin/nnUNetv2_train" ]; then
    NNUNET_TRAIN="${REPO_ROOT}/.venv/bin/nnUNetv2_train"
else
    NNUNET_TRAIN="nnUNetv2_train"
fi

echo "========================================================"
echo "Approach B — ROI variant cyst segmentation"
echo "  Variant  : ${VARIANT}"
echo "  Dataset  : ${DATASET_NAME}"
echo "  Config   : ${CONFIG}"
echo "  Fold     : ${FOLD}"
echo "========================================================"

if [ ! -d "${nnUNet_raw}/${DATASET_NAME}" ]; then
    echo "[ERROR] Dataset not found: ${nnUNet_raw}/${DATASET_NAME}"
    echo "        Run the prerequisite crop + prepare + preprocess steps first."
    exit 1
fi

# Guard: refuse to train if the custom single-fold split is missing, so we never
# silently fall back to nnUNet's random 5-fold CV (would leak val into training).
SPLITS="${nnUNet_preprocessed}/${DATASET_NAME}/splits_final.json"
if [ ! -f "${SPLITS}" ]; then
    echo "[ERROR] Missing ${SPLITS}"
    echo "        Run: python approach_b/write_splits_final.py --variant ${VARIANT}"
    echo "        (fold 0 must encode the project 247-train / 37-val split)."
    exit 1
fi

"${NNUNET_TRAIN}" "${DATASET_ID}" "${CONFIG}" "${FOLD}" --npz

echo "Training complete."
