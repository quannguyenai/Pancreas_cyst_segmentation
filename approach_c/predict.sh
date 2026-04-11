#!/usr/bin/env bash
# approach_c/predict.sh — Run inference with the fine-tuned approach_c model.
#
# Usage:
#   bash approach_c/predict.sh [CHECKPOINT] [SPLIT] [GPU_ID]
#
# Arguments:
#   CHECKPOINT  Path to .pth checkpoint (default: best_model.pth symlink)
#   SPLIT       train | val | test (default: test)
#   GPU_ID      GPU index (default: 0)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CHECKPOINT="${1:-${REPO_ROOT}/approach_c/checkpoints/best_model.pth}"
SPLIT="${2:-test}"
GPU="${3:-0}"
OUTPUT="${REPO_ROOT}/approach_c/predictions/${SPLIT}"

if [ ! -f "${CHECKPOINT}" ] && [ ! -L "${CHECKPOINT}" ]; then
    echo "[ERROR] Checkpoint not found: ${CHECKPOINT}"
    echo "        Train first: bash approach_c/train.sh"
    exit 1
fi

mkdir -p "${OUTPUT}"

echo "========================================================"
echo "Approach C — Inference"
echo "  Checkpoint : ${CHECKPOINT}"
echo "  Split      : ${SPLIT}"
echo "  Output     : ${OUTPUT}"
echo "========================================================"

python "${REPO_ROOT}/approach_c/inference.py" \
    --config     "${REPO_ROOT}/configs/paths.yaml" \
    --checkpoint "${CHECKPOINT}" \
    --split      "${SPLIT}" \
    --output     "${OUTPUT}" \
    --gpu        "${GPU}"
