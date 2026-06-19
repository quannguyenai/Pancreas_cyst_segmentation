#!/usr/bin/env bash
# approach_d/train.sh — Prepare dataset and train nnTransUNetTrainerV2 (fold 0).
#
# Prerequisites:
#   bash approach_d/setup.sh   (once per machine)
#
# Usage (from repo root):
#   bash approach_d/train.sh
#
# Auto-resumes if a checkpoint already exists.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── Activate isolated env ─────────────────────────────────────────────────────
# shellcheck disable=SC1091
source approach_d/set_env.sh

TASK_ID=1
TASK_NAME="PancreasCyst"
TRAINER="nnTransUNetTrainerV2"
CONFIG="3d_fullres"
FOLD=0

# ── Step 1: Build v1 dataset (fast — just symlinks + json/pkl) ────────────────
echo ""
echo "[1/3] Building nnUNet v1 dataset ..."
python approach_d/prepare_dataset.py --config configs/paths.yaml

# ── Step 2: Plan and preprocess ───────────────────────────────────────────────
PREPROC_DIR="${nnUNet_preprocessed}/Task$(printf '%03d' ${TASK_ID})_${TASK_NAME}"
PLANS_FILE="${PREPROC_DIR}/nnUNetPlans_plans_3D.pkl"

if [[ -f "${PLANS_FILE}" ]]; then
    echo ""
    echo "[2/3] Preprocessing already done, skipping."
else
    echo ""
    echo "[2/3] Running nnUNet_plan_and_preprocess ..."
    nnUNet_plan_and_preprocess -t "${TASK_ID}" --verify_dataset_integrity -tl 8 -tf 8
fi

# ── Step 3: Train ─────────────────────────────────────────────────────────────
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
echo "  Trainer  : ${TRAINER}"
echo "  Task     : Task$(printf '%03d' ${TASK_ID})_${TASK_NAME}"
echo "  Config   : ${CONFIG}"
echo "  Fold     : ${FOLD}"
echo "  Resume   : ${RESUME_FLAG:-no}"
echo "========================================================"

nnUNet_train "${CONFIG}" "${TRAINER}" "${TASK_ID}" "${FOLD}" ${RESUME_FLAG}
