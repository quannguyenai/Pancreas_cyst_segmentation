#!/usr/bin/env bash
# prepare_data.sh — Run all data preparation steps after placing images/masks.
#
# Usage (from repo root, venv active):
#   bash scripts/prepare_data.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Activate venv if not already active
if [[ -z "${VIRTUAL_ENV:-}" && -d ".venv" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# ── Verify data is present ────────────────────────────────────────────────────
IMAGE_COUNT=$(find data/images -name "*.nii.gz" 2>/dev/null | wc -l)
MASK_COUNT=$(find data/masks  -name "*.nii.gz" 2>/dev/null | wc -l)

if [[ "$IMAGE_COUNT" -eq 0 ]]; then
    echo "ERROR: No .nii.gz files found in data/images/"
    echo "Place your CT scans there first, then re-run this script."
    exit 1
fi
echo "Found ${IMAGE_COUNT} images and ${MASK_COUNT} masks."

# ── Step 1: Update split CSV paths to this machine ───────────────────────────
echo ""
echo "[1/3] Updating split CSV paths ..."
python data/prepare_dataset.py \
    --config configs/paths.yaml \
    --update-txts \
    --fix-cad-headers

# ── Step 2: Build nnUNet raw dataset ─────────────────────────────────────────
echo ""
echo "[2/3] Building nnUNet Dataset001_PancreasCyst ..."
python data/prepare_dataset.py \
    --config configs/paths.yaml \
    --build-nnunet

# ── Step 3: nnUNet plan and preprocess ───────────────────────────────────────
echo ""
echo "[3/3] Running nnUNet plan and preprocess (-np 8 workers) ..."
source scripts/set_nnunet_env.sh
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity -np 8

echo ""
echo "============================================================"
echo "  Data preparation complete. You can now train:"
echo ""
echo "  Approach A:  bash approach_a/train_mixed.sh 0"
echo "  Comparison:  python comparison/train.py --config configs/paths.yaml --mode 3d"
echo "============================================================"
