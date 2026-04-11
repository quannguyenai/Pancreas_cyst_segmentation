#!/usr/bin/env bash
# approach_b/predict.sh — Two-stage inference for the cascaded approach.
#
# Stage 1: Generate pancreas masks (if not already done).
# Stage 2: Predict cyst segmentation on cropped volumes.
# Stage 3: Paste predictions back into original image space.
#
# Usage:
#   bash approach_b/predict.sh [FOLD]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/set_nnunet_env.sh"

FOLD="${1:-0}"
CONFIG="3d_fullres"
DATASET_ID=10

CROPPED_INPUT="${REPO_ROOT}/approach_b/cropped/images"
CROPPED_PREDS="${REPO_ROOT}/approach_b/predictions/cropped"
FULL_PREDS="${REPO_ROOT}/approach_b/predictions/full_space"

mkdir -p "${CROPPED_PREDS}" "${FULL_PREDS}"

# ── Step 1: Ensure pancreas predictions exist ─────────────────────────────────
PANCREAS_PREDS="${REPO_ROOT}/approach_b/pancreas_preds"
if [ ! -d "${PANCREAS_PREDS}" ] || [ -z "$(ls -A "${PANCREAS_PREDS}" 2>/dev/null)" ]; then
    echo "[INFO] Pancreas predictions not found. Running Stage 1 ..."
    bash "${REPO_ROOT}/approach_b/stage1_pancreas_seg.sh"
fi

# ── Step 2: Crop test images using pancreas predictions ───────────────────────
CROP_STATS="${REPO_ROOT}/approach_b/crop_stats.json"
if [ ! -f "${CROP_STATS}" ]; then
    echo "[INFO] Cropping test images to pancreas ROI ..."
    python "${REPO_ROOT}/approach_b/crop_to_pancreas.py" \
        --config "${REPO_ROOT}/configs/paths.yaml" \
        --split test
fi

# ── Step 3: Format cropped test images for nnUNet ────────────────────────────
# Create a temporary imagesTs folder with _0000 suffix
TMP_INPUT=$(mktemp -d)
trap 'rm -rf "${TMP_INPUT}"' EXIT

for f in "${CROPPED_INPUT}"/*.nii.gz; do
    stem=$(basename "${f}" .nii.gz)
    ln -sf "$(realpath "${f}")" "${TMP_INPUT}/${stem}_0000.nii.gz"
done

# ── Step 4: nnUNet inference on cropped volumes ───────────────────────────────
echo "Running nnUNet inference on cropped volumes ..."
nnUNetv2_predict \
    -i "${TMP_INPUT}" \
    -o "${CROPPED_PREDS}" \
    -d "${DATASET_ID}" \
    -c "${CONFIG}" \
    -f "${FOLD}"

# ── Step 5: Paste predictions back into original image space ──────────────────
echo "Pasting predictions back into original space ..."
python "${REPO_ROOT}/approach_b/paste_back.py" \
    --config "${REPO_ROOT}/configs/paths.yaml" \
    --cropped-preds "${CROPPED_PREDS}" \
    --output "${FULL_PREDS}"

echo ""
echo "Full-space predictions: ${FULL_PREDS}"
