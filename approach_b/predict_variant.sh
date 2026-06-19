#!/usr/bin/env bash
# approach_b/predict_variant.sh — Inference for an Approach-B ROI redesign variant.
#
# Crops the test split to the fixed pancreas box (emitting the distance-map prior
# for priorchan), runs nnUNet on the cropped volumes, and pastes predictions back
# into original image space.
#
# Usage:
#   bash approach_b/predict_variant.sh <fixedbox|priorchan> [FOLD]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/set_nnunet_env.sh"

VARIANT="${1:?usage: predict_variant.sh <fixedbox|priorchan> [FOLD]}"
FOLD="${2:-0}"

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

eval "$("${PYTHON_BIN}" - "${VARIANT}" <<PY
import shlex, sys
sys.path.insert(0, "${REPO_ROOT}")
from configs import load_config
cfg = load_config("${REPO_ROOT}/configs/paths.yaml")
ab = cfg["approach_b"]
v = sys.argv[1]
with_prior = (v == "priorchan")
vals = {
    "CONFIG_NAME":       "3d_fullres",
    "DATASET_ID":        str(ab[f"{v}_dataset_id"]),
    "WITH_PRIOR":        "1" if with_prior else "0",
    "CROPPED_TEST_TXT":  ab["cropped_test_txt"],
    "CROPPED_PRIOR":     ab["cropped_prior"],
    "CROPPED_PREDS":     ab["predictions_cropped"],
    "FULL_PREDS":        ab["predictions_full"],
    "TEST_STATS":        ab["crop_stats_test_json"],
    "PANCREAS_PREDS":    ab["pancreas_preds"],
}
for k, val in vals.items():
    print(f"{k}={shlex.quote(str(val))}")
PY
)"

mkdir -p "${CROPPED_PREDS}" "${FULL_PREDS}"

# ── Step 1: Ensure pancreas predictions exist ─────────────────────────────────
if [ ! -d "${PANCREAS_PREDS}" ] || [ -z "$(ls -A "${PANCREAS_PREDS}" 2>/dev/null)" ]; then
    echo "[INFO] Pancreas predictions not found. Running Stage 1 ..."
    bash "${REPO_ROOT}/approach_b/stage1_pancreas_seg.sh"
fi

# ── Step 2: Crop test images to the fixed pancreas box ────────────────────────
EMIT_FLAG=()
[ "${WITH_PRIOR}" = "1" ] && EMIT_FLAG=(--emit-distance-channel)
echo "[INFO] Cropping test images (fixed-box${WITH_PRIOR:+, prior=${WITH_PRIOR}}) ..."
"${PYTHON_BIN}" "${REPO_ROOT}/approach_b/crop_to_pancreas.py" \
    --config "${REPO_ROOT}/configs/paths.yaml" \
    --split test \
    --mode fixed-box \
    --stats-path "${TEST_STATS}" \
    "${EMIT_FLAG[@]}"

"${PYTHON_BIN}" "${REPO_ROOT}/approach_b/prepare_cropped_dataset.py" \
    --config "${REPO_ROOT}/configs/paths.yaml" \
    --variant "${VARIANT}" \
    --skip-nnunet

# ── Step 3: Stage cropped test images for nnUNet (_0000 [+ _0001 prior]) ───────
TMP_INPUT=$(mktemp -d)
trap 'rm -rf "${TMP_INPUT}"' EXIT

while IFS= read -r line; do
    [ -z "${line}" ] && continue
    [ "${line}" = "image_path,mask_path" ] && continue
    img_path="${line%%,*}"
    stem="$(basename "${img_path}" .nii.gz)"
    ln -sf "$(realpath "${img_path}")" "${TMP_INPUT}/${stem}_0000.nii.gz"
    if [ "${WITH_PRIOR}" = "1" ]; then
        prior="${CROPPED_PRIOR}/${stem}.nii.gz"
        if [ ! -f "${prior}" ]; then
            echo "[ERROR] Missing prior crop for ${stem}: ${prior}"
            exit 1
        fi
        ln -sf "$(realpath "${prior}")" "${TMP_INPUT}/${stem}_0001.nii.gz"
    fi
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
