#!/usr/bin/env bash
# approach_c/train.sh — Fine-tune PanSegNet (pretrained) via nnUNet v1.
#
# Reuses approach_d's isolated venv + nnUNet v1 env.
# Shares the same Task001_PancreasCyst dataset as approach_d (no re-preprocessing).
# Checkpoints saved separately under nnTransUNetTrainerV2_Pretrained__nnUNetPlansv2.1/
#
# Usage (from repo root):
#   bash approach_c/train.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── Activate approach_d venv + nnUNet v1 env vars ─────────────────────────────
# shellcheck disable=SC1091
source approach_d/set_env.sh

# ── Pretrained weights config ─────────────────────────────────────────────────
export PANSEGNET_PRETRAINED_WEIGHTS="${REPO_ROOT}/pansegnet_weights/averaged_T1T2.model"
export PANSEGNET_WARMUP_EPOCHS=25

if [[ ! -f "${PANSEGNET_PRETRAINED_WEIGHTS}" ]]; then
    echo "[WARN] Pretrained weights not found at: ${PANSEGNET_PRETRAINED_WEIGHTS}"
    echo "       Training will proceed from scratch."
fi

TASK_ID=1
TASK_NAME="PancreasCyst"
TRAINER="nnTransUNetTrainerV2_Pretrained"
CONFIG="3d_fullres"
FOLD=0

# ── Install custom trainer into nnUNet package ────────────────────────────────
NNUNET_TRAINER_DIR="${REPO_ROOT}/approach_d/PaNSegNet/src/nnunet/training/network_training"
cp approach_c/nnTransUNetTrainerV2_Pretrained.py "${NNUNET_TRAINER_DIR}/"
echo "Custom trainer installed to ${NNUNET_TRAINER_DIR}/"

# ── Step 1: Build dataset (shared with approach_d — skips if already done) ───
echo ""
echo "[1/3] Building nnUNet v1 dataset ..."
python approach_d/prepare_dataset.py --config configs/paths.yaml

# ── Step 2: Plan and preprocess (skip if already done by approach_d) ─────────
PREPROC_DIR="${nnUNet_preprocessed}/Task$(printf '%03d' ${TASK_ID})_${TASK_NAME}"
PLANS_FILE="${PREPROC_DIR}/nnUNetPlans_plans_3D.pkl"

if [[ -f "${PLANS_FILE}" ]]; then
    echo ""
    echo "[2/3] Preprocessing already done (shared with approach_d), skipping."
else
    echo ""
    echo "[2/3] Running nnUNet_plan_and_preprocess ..."
    nnUNet_plan_and_preprocess -t "${TASK_ID}" -tl 8 -tf 8
fi

# ── Step 3: Train (auto-resume if checkpoint exists) ─────────────────────────
CKPT_DIR="${RESULTS_FOLDER}/nnUNet/${CONFIG}/Task$(printf '%03d' ${TASK_ID})_${TASK_NAME}/${TRAINER}__nnUNetPlansv2.1/fold_${FOLD}"
RESUME_FLAG=""
if [[ -f "${CKPT_DIR}/model_latest.model" ]]; then
    RESUME_FLAG="--continue_training"
    echo ""
    echo "[3/3] Checkpoint found — resuming training ..."

    # Back up epoch-600 snapshots before the extended run can overwrite them.
    # Each backup is written only once (skip if it already exists).
    for stem in model_best model_final_checkpoint; do
        src="${CKPT_DIR}/${stem}.model"
        dst="${CKPT_DIR}/${stem}_ep600.model"
        if [[ -f "${src}" && ! -f "${dst}" ]]; then
            cp "${src}"     "${dst}"
            cp "${src}.pkl" "${dst}.pkl"
            echo "  Backed up ${stem}.model → ${stem}_ep600.model"
        fi
    done
else
    echo ""
    echo "[3/3] Starting training from scratch ..."
fi

echo "========================================================"
echo "  Approach C — nnTransUNetTrainerV2_Pretrained"
echo "  Task     : Task$(printf '%03d' ${TASK_ID})_${TASK_NAME}"
echo "  Pretrained: ${PANSEGNET_PRETRAINED_WEIGHTS}"
echo "  Warmup   : ${PANSEGNET_WARMUP_EPOCHS} epochs frozen"
echo "  Resume   : ${RESUME_FLAG:-no}"
echo "========================================================"

nnUNet_train "${CONFIG}" "${TRAINER}" "${TASK_ID}" "${FOLD}" ${RESUME_FLAG}
