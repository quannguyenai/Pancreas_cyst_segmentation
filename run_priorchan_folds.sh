#!/usr/bin/env bash
# Launch B_priorchan (Dataset012_PriorChanCyst, 3d_fullres) 5-fold CV on GPUs 0-4.
# 2-channel input (MRI + pancreas signed-distance prior), fixed-box ROI.
# Launches fold 0 first and waits for nnU-Net to generate the 5-fold
# splits_final.json before launching folds 1-4 (avoids a generation race).
#
# Run inside container f449bb5e955f:
#   bash /workspace/team/pancrea_cyst_mri/run_priorchan_folds.sh

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

DATASET_ID=12
CONFIG=3d_fullres
SPLITS="${nnUNet_preprocessed}/Dataset012_PriorChanCyst/splits_final.json"

if [[ ! -d "${nnUNet_preprocessed}/Dataset012_PriorChanCyst/nnUNetPlans_${CONFIG}" ]]; then
    echo "[ERROR] Dataset012 not preprocessed."; exit 1
fi

echo "======================================================"
echo "B_priorchan — Dataset012_PriorChanCyst ${CONFIG}, 5-fold CV"
echo "nnUNetv2_train = $(which nnUNetv2_train)"
echo "======================================================"

# Fold 0 on GPU 0 — this generates splits_final.json (5-fold) if absent
echo "[GPU 0] fold 0 → logs/train_priorchan_fold0.log"
CUDA_VISIBLE_DEVICES=0 nohup nnUNetv2_train ${DATASET_ID} ${CONFIG} 0 --npz \
    > "${REPO}/logs/train_priorchan_fold0.log" 2>&1 &

# Wait for the 5-fold split file to appear before launching the rest
echo "Waiting for 5-fold splits_final.json ..."
until [[ -f "${SPLITS}" ]]; do sleep 3; done
sleep 5  # let the write settle
echo "splits_final.json ready."

# Folds 1-4 on GPUs 1-4
for FOLD in 1 2 3 4; do
    GPU=${FOLD}
    echo "[GPU ${GPU}] fold ${FOLD} → logs/train_priorchan_fold${FOLD}.log"
    CUDA_VISIBLE_DEVICES=${GPU} nohup nnUNetv2_train ${DATASET_ID} ${CONFIG} ${FOLD} --npz \
        > "${REPO}/logs/train_priorchan_fold${FOLD}.log" 2>&1 &
done

echo ""
echo "All 5 B_priorchan folds launched."
echo "Monitor: tail -f ${REPO}/logs/train_priorchan_fold0.log"
