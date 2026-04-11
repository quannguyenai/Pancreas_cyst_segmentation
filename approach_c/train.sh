#!/usr/bin/env bash
# approach_c/train.sh — Fine-tune PanSegNet for cyst segmentation.
#
# Prerequisites:
#   1. Place PanSegNet pretrained weights at:
#        approach_c/pretrained/PanSegNet.pth
#   2. Install MONAI: pip install monai>=1.5.2
#
# Usage:
#   bash approach_c/train.sh [GPU_ID]
#   bash approach_c/train.sh 1  # use GPU 1

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${1:-0}"

# ── Preflight check ───────────────────────────────────────────────────────────
WEIGHTS="${REPO_ROOT}/approach_c/pretrained/PanSegNet.pth"
if [ ! -f "${WEIGHTS}" ]; then
    echo "[WARN] PanSegNet weights not found at: ${WEIGHTS}"
    echo "       Training will proceed from scratch (random initialisation)."
    echo "       To use pretrained weights:"
    echo "         1. Obtain PanSegNet.pth from the original authors."
    echo "         2. Place at: ${WEIGHTS}"
fi

echo "========================================================"
echo "Approach C — Fine-tune PanSegNet for Cyst Segmentation"
echo "  Config : ${REPO_ROOT}/configs/paths.yaml"
echo "  GPU    : ${GPU}"
echo "========================================================"

python "${REPO_ROOT}/approach_c/finetune_trainer.py" \
    --config "${REPO_ROOT}/configs/paths.yaml" \
    --gpu "${GPU}" \
    "$@"
