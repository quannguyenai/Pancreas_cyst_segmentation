#!/usr/bin/env bash
# Ensemble each baseline's 5 CV folds on the held-out test set (74 cases) and
# build a comparison table — using the canonical eval_from_txt.py (mask_path
# column, mm spacing) so numbers are directly comparable to Approach A/B.
# Outputs go to results/baselines/ (NOT the stale results/comparison_table.csv).
#
# Run AFTER run_baselines.sh finishes, inside container f449bb5e955f:
#   bash /workspace/team/pancrea_cyst_mri/comparison/run_baselines_eval.sh
#
# Optional:  bash run_baselines_eval.sh "5 6 7"      # GPUs to spread models over
#            MODELS="unet_3d mednext" bash run_baselines_eval.sh

set -euo pipefail

REPO="/workspace/team/pancrea_cyst_mri"
VENV="${REPO}/comparison/.venv_baselines"
OUTDIR="${REPO}/results/baselines"
PRED_ROOT="${REPO}/comparison/predictions"

cd "${REPO}"
mkdir -p "${OUTDIR}" logs
export PATH="${VENV}/bin:${PATH}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1

MODELS=(${MODELS:-unet_3d unetpp transunet swinunetr mednext})
FOLDS=(${FOLDS:-0 1 2 3 4})

if [[ -n "${1:-}" ]]; then
    GPUS=($1)
else
    mapfile -t GPUS < <(nvidia-smi --query-gpu=index,memory.used \
        --format=csv,noheader,nounits | awk -F',' '($2+0)<2000{gsub(/ /,"",$1);print $1}')
fi
[[ ${#GPUS[@]} -eq 0 ]] && GPUS=(0)

ensemble_one() {  # $1=model  $2=gpu
    local m="$1" gpu="$2" cps=() f cp
    for f in "${FOLDS[@]}"; do
        cp="${REPO}/comparison/checkpoints/${m}/fold${f}/best_model.pth"
        [[ -f "${cp}" ]] || cp="${REPO}/comparison/checkpoints/${m}/fold${f}/final_model.pth"
        [[ -f "${cp}" ]] && cps+=("${cp}") || echo "[WARN] ${m} fold${f}: no checkpoint"
    done
    if [[ ${#cps[@]} -eq 0 ]]; then echo "[SKIP] ${m}: no checkpoints"; return; fi
    echo "[GPU ${gpu}] ensemble ${m} (${#cps[@]} folds) -> ${PRED_ROOT}/${m}"
    python comparison/test.py --config configs/paths.yaml --mode 3d --model "${m}" \
        --checkpoints "${cps[@]}" --split test --output "${PRED_ROOT}/${m}" \
        --per-fold-csv "${OUTDIR}/${m}_per_fold_test.csv" --gpu "${gpu}" \
        > "${REPO}/logs/eval_${m}.log" 2>&1
    # Canonical comparable metrics from the saved ensemble predictions.
    python comparison/eval_from_txt.py --split-txt data/test.txt \
        --pred-dir "${PRED_ROOT}/${m}" --name "${m}" \
        --out-prefix "${OUTDIR}/${m}_test" >> "${REPO}/logs/eval_${m}.log" 2>&1
    echo "[GPU ${gpu}] done ${m}"
}

# Run models in parallel, one per GPU (round-robin).
i=0
for m in "${MODELS[@]}"; do
    gpu="${GPUS[$(( i % ${#GPUS[@]} ))]}"
    ensemble_one "${m}" "${gpu}" &
    i=$((i+1))
    (( i % ${#GPUS[@]} == 0 )) && wait
done
wait

# Aggregate per-model summaries into one comparison table.
python - "${OUTDIR}" "${MODELS[@]}" <<'PY'
import sys, pandas as pd
from pathlib import Path
outdir = Path(sys.argv[1]); models = sys.argv[2:]
frames = []
for m in models:
    p = outdir / f"{m}_test_summary.csv"
    if p.exists():
        frames.append(pd.read_csv(p))
if frames:
    tbl = pd.concat(frames, ignore_index=True).sort_values("dice_mean", ascending=False)
    out = outdir / "baselines_comparison_table.csv"
    tbl.to_csv(out, index=False, float_format="%.4f")
    cols = ["approach","n","dice_mean","dice_median","dice_std","recall_mean","hd95_mean","n_total_miss"]
    print("\n==== Baseline 5-fold-ensemble test results (74 cases) ====")
    print(tbl[[c for c in cols if c in tbl.columns]].to_string(index=False))
    print(f"\nSaved: {out}")
else:
    print("[WARN] no per-model summaries found")
PY
echo "Done. See results/baselines/baselines_comparison_table.csv"
