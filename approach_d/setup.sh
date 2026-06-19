#!/usr/bin/env bash
# approach_d/setup.sh — One-time setup for PaNSegNet / nnTransUNetTrainerV2.
#
# Creates an isolated Python venv with nnUNet v1 + PaNSegNet inside approach_d/.
# Does NOT touch the root .venv or any existing approach setup.
#
# Usage (from repo root):
#   bash approach_d/setup.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APPROACH_D="${REPO_ROOT}/approach_d"
cd "$APPROACH_D"

# ── 1. Clone PaNSegNet ────────────────────────────────────────────────────────
if [[ -d "PaNSegNet/.git" ]]; then
    echo "[1/3] PaNSegNet already cloned, pulling latest ..."
    git -C PaNSegNet pull
else
    echo "[1/3] Cloning PaNSegNet ..."
    git clone https://github.com/NUBagciLab/PaNSegNet.git PaNSegNet
fi

# ── 2. Create isolated venv ───────────────────────────────────────────────────
if [[ ! -d ".venv" ]]; then
    echo "[2/3] Creating isolated venv ..."
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# ── 3. Install dependencies ───────────────────────────────────────────────────
echo "[3/3] Installing PyTorch + nnUNet v1 (PaNSegNet) ..."

# Detect CUDA version from nvcc or nvidia-smi. PyTorch only publishes wheels
# for a fixed set of tags; we clamp to the nearest supported one and honor a
# CUDA_TAG override so the caller can force-pick on unusual hosts.
SUPPORTED_TAGS=("cu118" "cu121" "cu124" "cu126" "cu128")
HIGHEST_TAG="${SUPPORTED_TAGS[-1]}"
CUDA_TAG="${CUDA_TAG:-}"
if [[ -z "$CUDA_TAG" ]]; then
    DETECTED="cu121"
    if command -v nvcc &>/dev/null; then
        CUDA_VER=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+" | head -1)
        MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
        MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
        DETECTED="cu${MAJOR}${MINOR}"
    fi
    if [[ " ${SUPPORTED_TAGS[*]} " == *" ${DETECTED} "* ]]; then
        CUDA_TAG="$DETECTED"
    else
        echo "  Detected ${DETECTED} has no PyTorch wheel; falling back to ${HIGHEST_TAG} (forward-compatible)."
        CUDA_TAG="$HIGHEST_TAG"
    fi
fi
echo "  CUDA tag: ${CUDA_TAG}"

pip install -q torch torchvision \
    --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

# nnUNet v1 ships inside PaNSegNet/src/
pip install -q -e PaNSegNet/src/

# Additional deps used by nnTransUNetTrainerV2
pip install -q einops timm

echo ""
echo "============================================================"
echo "  approach_d setup complete."
echo ""
echo "  Next step — prepare dataset and train:"
echo "    bash approach_d/train.sh"
echo "============================================================"
