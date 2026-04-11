#!/usr/bin/env bash
# approach_b/stage1_pancreas_seg.sh — Stage 1: Run pretrained PanSegNet to
# generate whole-pancreas masks for all images.
#
# The pancreas masks are used by crop_to_pancreas.py to define the ROI for
# Stage 2 cyst segmentation training/inference.
#
# ┌─ PREREQUISITE: PanSegNet weights ────────────────────────────────────────┐
# │ PanSegNet is a pretrained pancreas segmentation network.                 │
# │ Obtain the weights from the original authors or their repository, then   │
# │ place them in:                                                            │
# │   approach_b/pancreas_model/                                             │
# │                                                                           │
# │ Reference: https://github.com/mazurowski-lab/PanSegNet (if available)    │
# │ Contact the authors if weights are not publicly released.                │
# └───────────────────────────────────────────────────────────────────────────┘
#
# Usage:
#   bash approach_b/stage1_pancreas_seg.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${REPO_ROOT}/configs/paths.yaml"

MODEL_DIR="${REPO_ROOT}/approach_b/pancreas_model"
OUTPUT_DIR="${REPO_ROOT}/approach_b/pancreas_preds"

# ── Preflight checks ──────────────────────────────────────────────────────────
if [ ! -d "${MODEL_DIR}" ] || [ -z "$(ls -A "${MODEL_DIR}" 2>/dev/null)" ]; then
    echo "[ERROR] PanSegNet model weights not found at: ${MODEL_DIR}"
    echo ""
    echo "  1. Obtain weights from the PanSegNet authors."
    echo "  2. Place the model folder contents in: ${MODEL_DIR}"
    echo "  3. Re-run this script."
    exit 1
fi

INPUT_DIR=$(python3 -c "
import sys; sys.path.insert(0, '${REPO_ROOT}')
from configs import load_config
cfg = load_config('${CONFIG}')
print(cfg['data']['images'])
")

if [ ! -d "${INPUT_DIR}" ]; then
    echo "[ERROR] Images directory not found: ${INPUT_DIR}"
    echo "        Follow the data access instructions in README.md first."
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "========================================================"
echo "Approach B — Stage 1: PanSegNet Pancreas Segmentation"
echo "  Model  : ${MODEL_DIR}"
echo "  Input  : ${INPUT_DIR}"
echo "  Output : ${OUTPUT_DIR}"
echo "========================================================"

# Prepare a temporary imagesTs folder for nnUNet-style inference
TMP_INPUT=$(mktemp -d)
trap 'rm -rf "${TMP_INPUT}"' EXIT

for f in "${INPUT_DIR}"/*.nii.gz; do
    stem=$(basename "${f}" .nii.gz)
    ln -sf "$(realpath "${f}")" "${TMP_INPUT}/${stem}_0000.nii.gz"
done

# Run inference using nnUNet predict_from_modelfolder
# Adjust flags to match PanSegNet's actual nnUNet configuration
nnUNetv2_predict_from_modelfolder \
    -i "${TMP_INPUT}" \
    -o "${OUTPUT_DIR}" \
    -m "${MODEL_DIR}" \
    --save_probabilities \
    --continue_prediction

echo ""
echo "Pancreas predictions written to: ${OUTPUT_DIR}"
echo "Next: python approach_b/crop_to_pancreas.py --config configs/paths.yaml"
