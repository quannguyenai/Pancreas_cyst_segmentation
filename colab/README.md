# Colab T4 Training Guide

Files in this folder let you train Approach A (nnUNet) on **Google Colab T4 (15 GB VRAM)**
without modifying the main repository.

---

## Why a separate folder?

The auto-planned nnUNet config uses **patch [40, 192, 256] + batch 2 ≈ 14-18 GB VRAM** —
this exceeds T4's 15 GB and causes OOM. The fix reduces patch size to **[40, 128, 192]**
(2× fewer voxels → ~8-9 GB VRAM), which fits comfortably.

| | Default plans | T4 plans |
|---|---|---|
| Patch size | [40, 192, 256] | [40, 128, 192] |
| Batch size | 2 | 2 |
| Est. VRAM | ~17 GB ❌ | ~8 GB ✅ |
| Expected Dice drop | — | ~1-2 pts |

---

## Files

| File | Purpose |
|------|---------|
| `train_approach_a.ipynb` | Full Colab notebook — setup, data prep, training |
| `create_t4_plans.py` | Generates `nnUNetPlans_T4.json` with reduced patch size |

---

## Quick Start

1. **Upload dataset to Google Drive:**
   ```
   MyDrive/pancreas_cyst/data/images/*.nii.gz
   MyDrive/pancreas_cyst/data/masks/cyst_*.nii.gz
   ```

2. **Open `train_approach_a.ipynb` in Colab** (Runtime → Change runtime type → T4 GPU)

3. **Run all cells top to bottom.** Cells 5a–5c are skipped automatically on subsequent sessions.

4. **Session timeout handling:** Colab T4 sessions last ~12 hours. Training resumes automatically
   via `--c` because checkpoints are saved to Google Drive. Just re-run the notebook next session.

---

## Time estimate

1000 epochs × ~8-12 min/epoch on T4 = **130-200 hours total**  
≈ 11-17 Colab sessions of 12 hours each.

To finish faster, use a stronger GPU (A100 on Colab Pro, or Vast.ai/RunPod).

---

## Main repo commands (unchanged)

The `colab/` folder is fully self-contained. The main repo scripts
(`approach_a/train_mixed.sh`, etc.) are **not modified** and continue to work
as-is on any GPU with ≥ 20 GB VRAM.
