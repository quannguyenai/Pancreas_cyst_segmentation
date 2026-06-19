#!/usr/bin/env bash
# Launch 5-fold cross-validation training for Approach A 3D (Dataset001_PancreasCyst_3DA)
# across GPUs 0-4. ZScoreNormalization (correct for MRI).
#
# Prerequisites: preprocessing must be complete (run nnUNetv2_plan_and_preprocess first)
# Run from inside container f449bb5e955f:
#   bash /workspace/team/pancrea_cyst_mri/run_all_folds.sh

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

# Verify preprocessing is done
PREPROC_DIR="${nnUNet_preprocessed}/Dataset001_PancreasCyst_3DA/nnUNetPlans_3d_fullres"
if [[ ! -d "${PREPROC_DIR}" ]]; then
    echo "[ERROR] Preprocessing not complete. Wait for preprocess_d001.log to finish."
    echo "        Check: tail -f ${REPO}/logs/preprocess_d001.log"
    exit 1
fi

echo "======================================================"
echo "nnUNet_results  = ${nnUNet_results}"
echo "nnUNetv2_train  = $(which nnUNetv2_train)"
echo "Preprocessed OK : ${PREPROC_DIR}"
echo "======================================================"

# Approach A 3D — folds 0-4 on GPUs 0-4 (fresh 5-fold CV with ZScoreNorm)
for FOLD in 0 1 2 3 4; do
    GPU=${FOLD}
    LOG="${REPO}/logs/train_a3d_fold${FOLD}.log"
    echo "[GPU ${GPU}] Approach A 3D fold ${FOLD} → ${LOG}"
    CUDA_VISIBLE_DEVICES=${GPU} nohup bash "${REPO}/approach_a/train_mixed.sh" ${FOLD} \
        > "${LOG}" 2>&1 &
done

echo ""
echo "All 5 jobs launched (PIDs: $(jobs -p | tr '\n' ' '))"
echo "Monitor: tail -f ${REPO}/logs/train_a3d_fold0.log"
echo "GPU use: watch -n 30 nvidia-smi"
