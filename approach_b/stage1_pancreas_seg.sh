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

if [ -x "${REPO_ROOT}/.venv/bin/python3" ]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python3"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    PYTHON_BIN="$(command -v python)"
fi

"${PYTHON_BIN}" "${REPO_ROOT}/approach_b/run_pansegnet_inference.py" \
    --config "${CONFIG}" \
    "$@"
