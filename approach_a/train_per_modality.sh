#!/usr/bin/env bash
# approach_a/train_per_modality.sh — Approach A2: Train one nnUNet model per
# institution (AHN, CAD, EMC, IU, MCA, MCF, NYU, NU), using
# Dataset002_AHN … Dataset009_NU.
#
# Usage:
#   bash approach_a/train_per_modality.sh [FOLD]
#
# Each institution gets its own nnUNet dataset built by prepare_site_dataset.py,
# preprocessed, and trained in 3d_fullres configuration.
#
# Prerequisites:
#   Same as train_mixed.sh. All site datasets will be created automatically.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/set_nnunet_env.sh"

FOLD="${1:-0}"
CONFIG="3d_fullres"

# Institutions and corresponding dataset IDs (002–009)
SITES=(AHN CAD EMC IU MCA MCF NYU NU)
BASE_ID=2

echo "========================================================"
echo "Approach A2 — Per-institution model training"
echo "  Fold: ${FOLD} | Config: ${CONFIG}"
echo "  Institutions: ${SITES[*]}"
echo "========================================================"

for i in "${!SITES[@]}"; do
    SITE="${SITES[$i]}"
    DATASET_ID=$((BASE_ID + i))
    DATASET_NAME="Dataset$(printf '%03d' ${DATASET_ID})_${SITE}"

    echo ""
    echo "── Processing ${DATASET_NAME} ──"

    # Step 1: Build per-site nnUNet dataset
    python "${REPO_ROOT}/approach_a/prepare_site_dataset.py" \
        --config "${REPO_ROOT}/configs/paths.yaml" \
        --site   "${SITE}" \
        --dataset-id "${DATASET_ID}"

    # Step 2: Plan and preprocess
    nnUNetv2_plan_and_preprocess -d "${DATASET_ID}" \
        --verify_dataset_integrity -np 8

    # Step 3: Train
    nnUNetv2_train "${DATASET_ID}" "${CONFIG}" "${FOLD}" --npz

    echo "  → ${DATASET_NAME} training complete."
done

echo ""
echo "All per-institution models trained."
