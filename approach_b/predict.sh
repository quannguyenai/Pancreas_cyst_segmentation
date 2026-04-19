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
if [ -x "${REPO_ROOT}/.venv/bin/python3" ]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python3"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    PYTHON_BIN="$(command -v python)"
fi

if [ -x "${REPO_ROOT}/.venv/bin/nnUNetv2_predict" ]; then
    NNUNET_PREDICT="${REPO_ROOT}/.venv/bin/nnUNetv2_predict"
else
    NNUNET_PREDICT="nnUNetv2_predict"
fi

eval "$("${PYTHON_BIN}" - <<PY
import shlex, sys
sys.path.insert(0, "${REPO_ROOT}")
from configs import load_config
cfg = load_config("${REPO_ROOT}/configs/paths.yaml")
vals = {
    "CONFIG_NAME": "3d_fullres",
    "DATASET_ID": str(cfg["approach_b"]["nnunet_dataset_id"]),
    "CROPPED_INPUT": cfg["approach_b"]["cropped_images"],
    "CROPPED_TEST_TXT": cfg["approach_b"]["cropped_test_txt"],
    "CROPPED_PREDS": cfg["approach_b"]["predictions_cropped"],
    "FULL_PREDS": cfg["approach_b"]["predictions_full"],
    "TEST_STATS": cfg["approach_b"]["crop_stats_test_json"],
    "PANCREAS_PREDS": cfg["approach_b"]["pancreas_preds"],
}
for k, v in vals.items():
    print(f"{k}={shlex.quote(str(v))}")
PY
)"

mkdir -p "${CROPPED_PREDS}" "${FULL_PREDS}"

# ── Step 1: Ensure pancreas predictions exist ─────────────────────────────────
if [ ! -d "${PANCREAS_PREDS}" ] || [ -z "$(ls -A "${PANCREAS_PREDS}" 2>/dev/null)" ]; then
    echo "[INFO] Pancreas predictions not found. Running Stage 1 ..."
    bash "${REPO_ROOT}/approach_b/stage1_pancreas_seg.sh"
fi

# ── Step 2: Crop test images using pancreas predictions ───────────────────────
echo "[INFO] Cropping test images to pancreas ROI ..."
"${PYTHON_BIN}" "${REPO_ROOT}/approach_b/crop_to_pancreas.py" \
    --config "${REPO_ROOT}/configs/paths.yaml" \
    --split test \
    --stats-path "${TEST_STATS}"

"${PYTHON_BIN}" "${REPO_ROOT}/approach_b/prepare_cropped_dataset.py" \
    --config "${REPO_ROOT}/configs/paths.yaml" \
    --skip-nnunet

# ── Step 3: Format cropped test images for nnUNet ────────────────────────────
# Create a temporary imagesTs folder with _0000 suffix
TMP_INPUT=$(mktemp -d)
trap 'rm -rf "${TMP_INPUT}"' EXIT

while IFS= read -r line; do
    [ -z "${line}" ] && continue
    if [ "${line}" = "image_path,mask_path" ]; then
        continue
    fi
    img_path="${line%%,*}"
    stem="$(basename "${img_path}" .nii.gz)"
    ln -sf "$(realpath "${img_path}")" "${TMP_INPUT}/${stem}_0000.nii.gz"
done < "${CROPPED_TEST_TXT}"

# ── Step 4: nnUNet inference on cropped volumes ───────────────────────────────
echo "Running nnUNet inference on cropped volumes ..."
"${NNUNET_PREDICT}" \
    -i "${TMP_INPUT}" \
    -o "${CROPPED_PREDS}" \
    -d "${DATASET_ID}" \
    -c "${CONFIG_NAME}" \
    -f "${FOLD}"

# ── Step 5: Paste predictions back into original image space ──────────────────
echo "Pasting predictions back into original space ..."
"${PYTHON_BIN}" "${REPO_ROOT}/approach_b/paste_back.py" \
    --config "${REPO_ROOT}/configs/paths.yaml" \
    --crop-stats "${TEST_STATS}" \
    --cropped-preds "${CROPPED_PREDS}" \
    --output "${FULL_PREDS}"

echo ""
echo "Full-space predictions: ${FULL_PREDS}"
