#!/usr/bin/env bash
# setup_gpu.sh — One-shot environment setup on a fresh GPU machine.
#
# Usage (from repo root):
#   bash scripts/setup_gpu.sh          # interactive, asks about CUDA version
#   CUDA=cu126 bash scripts/setup_gpu.sh   # non-interactive
#
# After this script finishes, all that remains is:
#   1. Place your data in data/images/ and data/masks/
#   2. Run: bash scripts/prepare_data.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── CUDA version ──────────────────────────────────────────────────────────────
CUDA="${CUDA:-}"
if [[ -z "$CUDA" ]]; then
    echo ""
    echo "Select your CUDA version:"
    echo "  1) cu126  (CUDA 12.6 — RTX 4090/5090, A100, H100)"
    echo "  2) cu121  (CUDA 12.1)"
    echo "  3) cu118  (CUDA 11.8 — older GPUs)"
    echo "  4) cpu    (CPU only / no GPU)"
    read -rp "Choice [1]: " choice
    case "${choice:-1}" in
        1) CUDA=cu126 ;;
        2) CUDA=cu121 ;;
        3) CUDA=cu118 ;;
        4) CUDA=cpu   ;;
        *) CUDA=cu126 ;;
    esac
fi
echo "Using PyTorch index: ${CUDA}"

# ── Virtual environment ────────────────────────────────────────────────────────
if [[ ! -d ".venv" ]]; then
    echo ""
    echo "Creating .venv ..."
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# ── PyTorch ───────────────────────────────────────────────────────────────────
echo ""
echo "Installing PyTorch (${CUDA}) ..."
if [[ "$CUDA" == "cpu" ]]; then
    pip install -q torch torchvision
else
    pip install -q torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA}"
fi

# ── Remaining dependencies ────────────────────────────────────────────────────
echo ""
echo "Installing requirements.txt ..."
pip install -q -r requirements.txt

# ── nnUNet environment variables ──────────────────────────────────────────────
echo ""
echo "Exporting nnUNet environment variables ..."
source scripts/set_nnunet_env.sh

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Setup complete."
echo ""
echo "  Next steps:"
echo ""
echo "  1. Place data files:"
echo "       data/images/*.nii.gz   (CT scans)"
echo "       data/masks/*.nii.gz    (cyst masks)"
echo ""
echo "  2. Prepare dataset (update paths + build nnUNet raw):"
echo "       source .venv/bin/activate"
echo "       bash scripts/prepare_data.sh"
echo ""
echo "  3. Train:"
echo "       # Approach A (nnUNet — recommended):"
echo "       bash approach_a/train_mixed.sh 0"
echo ""
echo "       # Comparison baselines (2D/3D):"
echo "       python comparison/train.py --config configs/paths.yaml --mode 3d --model vnet"
echo ""
echo "  Activate the venv in future sessions with:"
echo "       source .venv/bin/activate && source scripts/set_nnunet_env.sh"
echo "============================================================"
