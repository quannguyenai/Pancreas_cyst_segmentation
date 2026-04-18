#!/usr/bin/env bash
# approach_d/set_env.sh — Activate approach_d venv and export nnUNet v1 env vars.
#
# Source this file (do not execute):
#   source approach_d/set_env.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APPROACH_D="${REPO_ROOT}/approach_d"

# Activate isolated venv
if [[ -f "${APPROACH_D}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${APPROACH_D}/.venv/bin/activate"
else
    echo "ERROR: approach_d/.venv not found. Run: bash approach_d/setup.sh"
    return 1
fi

# nnUNet v1 uses different env var names than v2
export nnUNet_raw_data_base="${APPROACH_D}/nnunet_v1/raw"
export nnUNet_preprocessed="${APPROACH_D}/nnunet_v1/preprocessed"
export RESULTS_FOLDER="${APPROACH_D}/nnunet_v1/results"

mkdir -p "${nnUNet_raw_data_base}/nnUNet_raw_data"
mkdir -p "${nnUNet_preprocessed}"
mkdir -p "${RESULTS_FOLDER}"

echo "approach_d env active:"
echo "  nnUNet_raw_data_base = ${nnUNet_raw_data_base}"
echo "  nnUNet_preprocessed  = ${nnUNet_preprocessed}"
echo "  RESULTS_FOLDER       = ${RESULTS_FOLDER}"
