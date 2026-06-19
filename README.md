# Pancreatic Cyst Segmentation

**Multi-approach benchmark for automated pancreatic cyst segmentation in T2 MRI.**

This repository implements and compares four segmentation strategies on a
multi-institutional T2 MRI dataset (358 cases, 8 sites).

---

## Quick Start on setting up a new machine

```bash
# 1. Clone and enter repo
git clone git@github.com:quannguyenai/Pancreas_cyst_segmentation.git
cd Pancreas_cyst_segmentation

# 2. Install everything (creates .venv, installs PyTorch + all deps)
bash scripts/setup_gpu.sh

# 3. Place dataset files
#    data/images/EMC024.nii.gz          (358 T2 MRI scans)
#    data/masks/cyst_emc_024.nii.gz     (358 binary cyst masks)

# 4. Prepare data
source .venv/bin/activate
bash scripts/prepare_data.sh

# 5. Train (Approach A 3D — recommended)
bash approach_a/train_mixed.sh 0
```

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Repository Structure](#repository-structure)
4. [Installation](#installation)
5. [Data Preparation](#data-preparation)
6. [Approach A — nnUNet (3D and 2.5D)](#approach-a)
7. [Approach B — Cascaded Pipeline](#approach-b)
8. [Approach D — nnUNet v1 + PanSegNet](#approach-d)
9. [Evaluation & Visualisation](#evaluation--visualisation)
10. [Results](#results)

---

## Overview

Pancreatic cysts are fluid-filled lesions with variable malignant potential.
Automated segmentation from T2 MRI is clinically valuable for surveillance and
surgical planning. This repository benchmarks four deep-learning segmentation
strategies:

| Approach | Method | Key detail |
|----------|--------|------------|
| **A 3D** | nnUNet v2 — 3D full-resolution | Best overall Dice (0.674) |
| **A 2.5D** | nnUNet v2 — 2.5D stack-of-5-slices | 5-channel 2D input, each channel = adjacent axial slice |
| **B** | Cascaded: PanSegNet pancreas → crop → nnUNet cyst | Pancreas ROI reduces search space |
| **D** | nnUNet v1 + PanSegNet weights | Separate environment, own `.venv` |

---

## Dataset

| Statistic | Value |
|-----------|-------|
| Total cases | 358 T2 MRI volumes |
| Institutions | 8 (AHN, CAD, EMC, IU, MCA, MCF, NU, NYU) |
| Training split | 247 cases (`data/train.txt`) |
| Validation split | 37 cases (`data/val.txt`) |
| Test split | 74 cases (`data/test.txt`) |
| Image format | NIfTI (`.nii.gz`) |
| Annotation | Binary cyst masks |
| Mask naming | `cyst_<site>_<id>.nii.gz` (e.g. `cyst_emc_024.nii.gz`) |

> **Data access:** Images and masks are not publicly distributed due to
> institutional data agreements. Contact the dataset curators for access.
>
> Once granted, place files as:
> ```
> data/images/EMC024.nii.gz
> data/masks/cyst_emc_024.nii.gz
> ```

---

## Repository Structure

```
pancrea_cyst/
├── README.md
├── requirements.txt
├── configs/
│   ├── paths.yaml                  # Central path config (edit PANCREAS_CYST_ROOT)
│   └── __init__.py                 # load_config() helper
│
├── data/
│   ├── prepare_dataset.py          # NIfTI → nnUNet format; fix CAD affines
│   ├── train.txt / val.txt / test.txt / all_train.txt
│   ├── images/                     # Raw T2 MRI scans (not in git)
│   └── masks/                      # Binary GT masks (not in git)
│
├── approach_a/                     # nnUNet v2 — 3D and 2.5D
│   ├── train_mixed.sh              # 3D full-resolution training
│   ├── train_stack5.sh             # 2.5D stack-of-5 training
│   ├── predict.sh                  # 3D inference → approach_a/prediction/3d_fullres/
│   ├── predict_stack5.sh           # 2.5D inference → approach_a/prediction/2d_stack5/
│   ├── prepare_stack5_dataset.py   # Build 5-channel 2.5D nnUNet dataset
│   └── prepare_site_dataset.py     # Build per-institution datasets
│
├── approach_b/                     # Cascaded: pancreas → crop → cyst
│   ├── run_pansegnet_inference.py  # Stage 1: pancreas masks via PanSegNet
│   ├── crop_to_pancreas.py         # Stage 2: crop volumes to pancreas ROI
│   ├── paste_back.py               # Restore predictions to full-space coords
│   ├── prepare_cropped_dataset.py  # Build nnUNet dataset from cropped volumes
│   ├── train.sh / predict.sh
│   └── predictio/full_space/       # Final full-space test predictions
│
├── approach_c/                     # Fine-tune PanSegNet (experimental)
│   ├── finetune_trainer.py
│   ├── pansegnet.py
│   └── pretrained/                 # Place PanSegNet.pth here
│
├── approach_d/                     # nnUNet v1 + PanSegNet (own .venv)
│   ├── setup.sh                    # Install nnUNet v1 into approach_d/.venv
│   ├── set_env.sh                  # Export nnUNet v1 env vars
│   ├── prepare_dataset.py
│   ├── train.sh / predict.sh
│   ├── evaluate.py
│   └── prediction/                 # Test predictions (authoritative folder)
│
├── baseline/                       # Semi-supervised baselines (preserved)
│   ├── 2D-UNet/
│   ├── 3D-VNet/
│   └── environment.yaml            # Conda env for baseline models
│
├── comparison/                     # 2D U-Net / 3D V-Net baseline runners
│   ├── train.py / test.py / evaluate.py
│   ├── networks/                   # unet.py, VNet.py, unetr.py
│   └── dataloaders/
│
├── pansegnet_weights/              # Pre-trained PanSegNet weights
│   ├── Task110_PancreasT1MRI/
│   ├── Task111_PancreasT2MRI/
│   └── averaged_T1T2.model
│
├── nnUnet/
│   ├── nnUNet_raw/                 # nnUNet-format datasets (Dataset001, Dataset010, 011)
│   ├── nnUNet_preprocessed/        # Preprocessed patches (generated; large)
│   └── nnUNet_results/             # Trained checkpoints + validation predictions
│       ├── Dataset001_PancreasCyst_3DA/   # Approach A 3D model
│       ├── Dataset011_PancreasCyst25D/    # Approach A 2.5D model
│       └── Dataset010_CroppedCyst/        # Approach B cropped model
│
├── scripts/
│   ├── setup_gpu.sh                # One-shot env setup
│   ├── prepare_data.sh             # Full data prep pipeline
│   ├── set_nnunet_env.sh           # Export NNUNET_* env vars
│   └── eda_pancrea_cyst_dataset.py # Dataset statistics and EDA
│
├── eda/                            # Exploratory data analysis outputs
│   ├── eda_summary.md
│   ├── case_level_eda.csv
│   └── *.png                       # Distribution plots
│
├── colab/                          # Google Colab training notebooks
│   ├── train_approach_a.ipynb
│   └── create_t4_plans.py          # Adapt plans for T4 GPU memory
│
└── results/
    ├── comparison_by_case.csv      # Per-case metrics for all 4 approaches
    │                               # (74 test cases × Dice/precision/recall/F1/HD95/ASD
    │                               #  + volumes, site grouping)
    ├── comparison_table.csv        # Summary table
    ├── per_case/                   # Per-case CSVs by approach
    ├── figures/                    # All output figures
    │   ├── heatmap_dice.png        # Per-case Dice heatmap grouped by site
    │   ├── overview_sorted.png     # Per-case Dice line plot
    │   ├── site_summary.png        # Per-site mean Dice + case counts
    │   ├── overview_summary.png    # Bar chart mean ± std all approaches
    │   ├── top5_best.png           # Segmentation overlays, top-5 best cases
    │   ├── top5_worst.png          # Segmentation overlays, top-5 worst cases
    │   ├── gradcam_ct/             # Grad-CAM figures (CT normalization)
    │   └── gradcam_mri/            # Grad-CAM figures (MRI normalization)
    ├── gradcam_report.pdf          # Sectioned Grad-CAM PDF (CT norm)
    ├── gradcam_report_mri.pdf      # Sectioned Grad-CAM PDF (MRI norm)
    ├── gradcam.py                  # Grad-CAM script (usage: python gradcam.py [ct|mri])
    ├── merge_gradcam_pdf.py        # PDF builder (usage: python merge_gradcam_pdf.py [ct|mri])
    ├── plot_results.py             # Metric heatmaps + overview plots
    ├── visualize_cases.py          # Segmentation overlay figures
    ├── visualize_activations.py    # Probability heatmap figures
    └── visualize_prob_heatmaps.py  # Per-case heatmap grid
```

---

## Installation

### Approaches A, B, C (nnUNet v2)

Requires Python ≥ 3.10.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

### Approach D (nnUNet v1, isolated environment)

```bash
cd approach_d
bash setup.sh          # creates approach_d/.venv with nnUNet v1
source set_env.sh      # exports NNUNET_* paths
```

### Baseline (2D U-Net / 3D V-Net)

```bash
conda env create -f baseline/environment.yaml
conda activate monai
```

---

## Data Preparation

```bash
# Fix CAD mask affine headers and update split CSV paths
python data/prepare_dataset.py --config configs/paths.yaml \
    --fix-cad-headers --update-txts

# Build nnUNet Dataset001 (3D) and Dataset011 (2.5D)
python data/prepare_dataset.py --config configs/paths.yaml --build-nnunet
python approach_a/prepare_stack5_dataset.py

# Plan and preprocess
source scripts/set_nnunet_env.sh
nnUNetv2_plan_and_preprocess -d 1  --verify_dataset_integrity -np 8
nnUNetv2_plan_and_preprocess -d 11 --verify_dataset_integrity -np 8
```

---

## Approach A

**nnUNet v2** trained directly on the full multi-institutional T2 MRI dataset.

### A 3D — 3D Full-Resolution

```bash
source scripts/set_nnunet_env.sh
bash approach_a/train_mixed.sh 0          # fold 0, ~1000 epochs
bash approach_a/predict.sh "0"            # → approach_a/prediction/3d_fullres/
```

Checkpoint: `nnUnet/nnUNet_results/Dataset001_PancreasCyst_3DA/…/fold_0/checkpoint_final.pth`

### A 2.5D — Stack-of-5 Slices

Each 3D case is rewritten as a 5-channel 2D dataset (axial slices ±2 neighbours
as channels), then trained with nnUNet's built-in `2d` configuration.

```bash
python approach_a/prepare_stack5_dataset.py
nnUNetv2_plan_and_preprocess -d 11 --verify_dataset_integrity
bash approach_a/train_stack5.sh 0
bash approach_a/predict_stack5.sh "0"     # → approach_a/prediction/2d_stack5/
```

---

## Approach B

**Cascaded pipeline:** PanSegNet generates a pancreas mask → volumes are cropped
to the pancreas ROI → nnUNet trains on the cropped sub-volumes.

```bash
# Stage 1: pancreas segmentation
bash approach_b/stage1_pancreas_seg.sh

# Stage 2: crop to ROI
python approach_b/crop_to_pancreas.py --config configs/paths.yaml

# Build and preprocess cropped dataset (Dataset010)
python approach_b/prepare_cropped_dataset.py
source scripts/set_nnunet_env.sh
nnUNetv2_plan_and_preprocess -d 10 --verify_dataset_integrity

# Train and predict
bash approach_b/train.sh 0
bash approach_b/predict.sh 0              # → approach_b/predictio/full_space/
```

---

## Approach D

**nnUNet v1** with PanSegNet pretrained weights. Uses its own isolated virtual
environment under `approach_d/.venv`.

```bash
cd approach_d
source set_env.sh
bash train.sh
bash predict.sh                           # → approach_d/prediction/
```

---

## Evaluation & Visualisation

All scripts run from the project root with `.venv` active.

```bash
# Metric heatmaps grouped by site (Dice, HD95, ASD)
python results/plot_results.py

# Segmentation overlay figures — top-5 best/worst cases
python results/visualize_cases.py

# Probability heatmap figures
python results/visualize_activations.py

# Grad-CAM saliency maps (ct = matches training norm, mri = per-case z-score)
python results/gradcam.py ct              # → results/figures/gradcam_ct/
python results/gradcam.py mri             # → results/figures/gradcam_mri/

# Build sectioned PDF report
python results/merge_gradcam_pdf.py ct    # → results/gradcam_report.pdf
python results/merge_gradcam_pdf.py mri   # → results/gradcam_report_mri.pdf
```

The comparison CSV at `results/comparison_by_case.csv` contains per-case
Dice, precision, recall, F1, HD95, ASD, and physical volumes (mm³) for all
74 test cases across all four approaches.

---

## Results

Test-set performance (74 cases, 8 institutions):

| Approach | Dice ↑ | HD95 ↓ | ASD ↓ (mm) |
|----------|--------|--------|------------|
| **A 3D** — nnUNet v2 3D | **0.674 ± 0.256** | **0.571** | **16.1** |
| D — nnUNet v1 | 0.643 ± 0.272 | 0.539 | 17.7 |
| B — Cascaded | 0.614 ± 0.279 | 0.494 | 19.7 |
| A 2.5D — nnUNet v2 2.5D | 0.594 ± 0.275 | 0.479 | 19.7 |

Per-site mean Dice (A 3D): MCF 0.807 > CAD 0.789 > AHN 0.772 > MCA 0.696 > IU 0.651 > EMC 0.626 > NYU 0.618 > NU 0.591

> Detailed per-case breakdown, site-level analysis, and Grad-CAM saliency
> visualisations are in `results/figures/` and the PDF reports.
