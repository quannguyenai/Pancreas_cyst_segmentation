#!/usr/bin/env bash
# Train the comparison architecture baselines with 5-fold cross-validation,
# distributing the 25 runs (5 models x 5 folds) across idle GPUs via a
# worker-pool queue: one training job per GPU at a time, each GPU pulling the
# next job as soon as it frees up — so all idle GPUs stay saturated.
#
# Run from inside container f449bb5e955f:
#   bash /workspace/team/pancrea_cyst_mri/comparison/run_baselines.sh
#
# Optional:
#   bash run_baselines.sh "5 6 7"                # restrict to these GPUs
#   MODELS="unet_3d mednext" FOLDS="0 1" bash run_baselines.sh
#   MAX_EPOCH=300 BATCH=2 LR=0.01 bash run_baselines.sh
# Default GPUs = all currently-idle GPUs (mem.used < 2 GB).

set -euo pipefail

REPO="/workspace/team/pancrea_cyst_mri"
VENV="${REPO}/comparison/.venv_baselines"

cd "${REPO}"
mkdir -p logs
export PATH="${VENV}/bin:${PATH}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1

MODELS=(${MODELS:-unet_3d unetpp transunet swinunetr mednext})
FOLDS=(${FOLDS:-0 1 2 3 4})
MAX_EPOCH="${MAX_EPOCH:-300}"
BATCH="${BATCH:-2}"
LR="${LR:-0.01}"

# GPU list: arg 1 overrides; else auto-detect idle GPUs.
if [[ -n "${1:-}" ]]; then
    GPUS=($1)
else
    mapfile -t GPUS < <(nvidia-smi --query-gpu=index,memory.used \
        --format=csv,noheader,nounits | awk -F',' '($2+0)<2000{gsub(/ /,"",$1);print $1}')
fi
if [[ ${#GPUS[@]} -eq 0 ]]; then echo "[ERROR] no idle GPUs found"; exit 1; fi

# Ensure the 5-fold split files exist (idempotent).
if [[ ! -f comparison/splits/fold0_train.txt ]]; then
    echo "[setup] generating 5-fold splits ..."
    python comparison/make_cv_splits.py --config configs/paths.yaml
fi

# Build the job queue: one line per (model, fold).
QUEUE="$(mktemp)"; LOCK="${QUEUE}.lock"; : > "${LOCK}"
for m in "${MODELS[@]}"; do for f in "${FOLDS[@]}"; do echo "${m} ${f}"; done; done > "${QUEUE}"
NJOBS=$(wc -l < "${QUEUE}")

echo "======================================================"
echo "Comparison baselines — 5-fold CV (3D, patch 96^3)"
echo "models = ${MODELS[*]}"
echo "folds  = ${FOLDS[*]}"
echo "GPUs   = ${GPUS[*]}   jobs=${NJOBS}   epochs=${MAX_EPOCH} batch=${BATCH} lr=${LR}"
echo "======================================================"

pop_job() {  # atomically pop one line from the queue; echoes "" when empty
    exec 9>"${LOCK}"; flock 9
    local line; line="$(head -n1 "${QUEUE}")"
    [[ -n "${line}" ]] && sed -i '1d' "${QUEUE}"
    flock -u 9
    echo "${line}"
}

worker() {   # $1 = physical GPU id; drains the queue, one job at a time
    local gpu="$1" job m f LOG
    while true; do
        job="$(pop_job)"; [[ -z "${job}" ]] && break
        read -r m f <<< "${job}"
        LOG="${REPO}/logs/baseline_${m}_fold${f}.log"
        echo "[GPU ${gpu}] start ${m} fold${f} -> ${LOG}"
        # train.py pins CUDA_VISIBLE_DEVICES from --gpu; pass the physical id.
        python comparison/train.py --config configs/paths.yaml --mode 3d \
            --model "${m}" --fold "${f}" --exp "${m}" \
            --max-epoch "${MAX_EPOCH}" --batchsize "${BATCH}" --base-lr "${LR}" \
            --gpu "${gpu}" > "${LOG}" 2>&1 \
            && echo "[GPU ${gpu}] done  ${m} fold${f}" \
            || echo "[GPU ${gpu}] FAIL  ${m} fold${f} (see ${LOG})"
    done
    echo "[GPU ${gpu}] queue drained"
}

for g in "${GPUS[@]}"; do worker "${g}" & done
wait
rm -f "${QUEUE}" "${LOCK}"
echo "All 5-fold CV training complete. Checkpoints: comparison/checkpoints/<model>/fold<k>/best_model.pth"
echo "Next: bash comparison/run_baselines_eval.sh"
