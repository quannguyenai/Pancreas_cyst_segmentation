#!/usr/bin/env bash
# Launch Approach B (Dataset010_CroppedCyst, 3d_fullres) cross-validation folds.
# Default: folds 0 1 2 on GPUs 5 6 7 (the idle ones while Approach A uses 0-4).
# Override folds/GPUs via args: bash run_b_folds.sh "3 4" "5 6"
#
# Run from inside container f449bb5e955f:
#   bash /workspace/team/pancrea_cyst_mri/run_b_folds.sh

set -euo pipefail

REPO="/workspace/team/pancrea_cyst_mri"
VENV="/workspace/team/pancreas_ct_cyst_seg/.venv"

cd "${REPO}"
mkdir -p logs

export PATH="${VENV}/bin:${PATH}"
export nnUNet_raw="${REPO}/nnUnet/nnUNet_raw"
export nnUNet_preprocessed="${REPO}/nnUnet/nnUNet_preprocessed"
export nnUNet_results="${REPO}/nnUnet/nnUNet_results"
export nnUNet_compile=0
export OMP_NUM_THREADS=1

DATASET_ID=10
CONFIG=3d_fullres
FOLDS=(${1:-0 1 2})
GPUS=(${2:-5 6 7})

PREPROC_DIR="${nnUNet_preprocessed}/Dataset010_CroppedCyst/nnUNetPlans_${CONFIG}"
if [[ ! -d "${PREPROC_DIR}" ]]; then
    echo "[ERROR] Preprocessing not complete: ${PREPROC_DIR}"
    exit 1
fi

echo "======================================================"
echo "Approach B — Dataset010_CroppedCyst ${CONFIG}"
echo "nnUNetv2_train = $(which nnUNetv2_train)"
echo "Folds          = ${FOLDS[*]}   GPUs = ${GPUS[*]}"
echo "======================================================"

for i in "${!FOLDS[@]}"; do
    FOLD=${FOLDS[$i]}
    GPU=${GPUS[$i]}
    LOG="${REPO}/logs/train_b_fold${FOLD}.log"
    echo "[GPU ${GPU}] Approach B fold ${FOLD} → ${LOG}"
    CUDA_VISIBLE_DEVICES=${GPU} nohup nnUNetv2_train "${DATASET_ID}" "${CONFIG}" "${FOLD}" --npz \
        > "${LOG}" 2>&1 &
done

echo ""
echo "Launched (PIDs: $(jobs -p | tr '\n' ' '))"
echo "Monitor: tail -f ${REPO}/logs/train_b_fold0.log"
