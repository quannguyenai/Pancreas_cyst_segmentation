# Pancreatic Cyst Segmentation

**Multi-approach benchmark for automated pancreatic cyst segmentation in CT.**

This repository implements and compares four segmentation strategies on a
multi-institutional CT dataset (358 cases, 8 sites). The work is part of an
ongoing research project in medical image analysis.

---

## Quick Start on a Fresh GPU Machine

```bash
# 1. Clone and enter repo
git clone git@github.com:quannguyenai/Pancreas_cyst_segmentation.git
cd Pancreas_cyst_segmentation

# 2. Install everything (creates .venv, installs PyTorch + all deps)
bash scripts/setup_gpu.sh

# 3. Place dataset files
#    data/images/EMC024.nii.gz  …  (358 CT scans)
#    data/masks/cyst_emc_024.nii.gz  …  (358 masks)

# 4. Prepare data (update paths, build nnUNet raw, preprocess)
source .venv/bin/activate
bash scripts/prepare_data.sh

# 5. Train (Approach A — nnUNet, recommended)
bash approach_a/train_mixed.sh 0
```

That is everything. Steps 2 and 4 are one-time; subsequent training sessions
only need `source .venv/bin/activate && source scripts/set_nnunet_env.sh`.

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Repository Structure](#repository-structure)
4. [Installation](#installation)
5. [Data Preparation](#data-preparation)
6. [Approach A — Direct nnUNet Baseline](#approach-a)
7. [Approach B — Cascaded Pipeline](#approach-b)
8. [Approach C — Fine-tuned PanSegNet](#approach-c)
9. [Comparison Baselines](#comparison-baselines)
10. [Evaluation](#evaluation)
11. [Results](#results)
12. [Citation](#citation)
13. [License](#license)

---

## Overview

Pancreatic cysts are fluid-filled lesions in the pancreas with variable
malignant potential. Automated segmentation from CT scans is clinically
valuable for surveillance and surgical planning. This repository benchmarks
three complementary deep-learning segmentation strategies:

| Approach | Method | Key Component |
|----------|--------|---------------|
| **A** | Direct nnUNet v2 baseline | Train on full dataset or per-institution splits |
| **B** | Cascaded: pancreas → crop → cyst | PanSegNet stage-1 pancreas mask guides ROI |
| **C** | Fine-tune PanSegNet for cyst | Transfer encoder, retrain segmentation head |
| **Comparison** | 2D U-Net / 3D V-Net baselines | Semi-supervised baselines with BCP |

---

## Dataset

| Statistic | Value |
|-----------|-------|
| Total cases | 358 CT volumes |
| Institutions | 8 (AHN, CAD, EMC, IU, MCA, MCF, NYU, NU) |
| Training split | 247 cases (`data/train.txt`) |
| Validation split | 37 cases (`data/val.txt`) |
| Test split | 74 cases (`data/test.txt`) |
| Image format | NIfTI (.nii.gz) |
| Annotation | Binary cyst masks |

> **Data access:** The CT images and segmentation masks are not publicly
> distributed in this repository due to institutional data agreements.
> To request access, contact the dataset curators and follow the data
> sharing protocol described in the associated paper.
>
> Once access is granted, place files as follows:
> ```
> data/images/EMC024.nii.gz   # CT scans
> data/masks/cyst_emc_024.nii.gz  # cyst masks
> ```

---

## Repository Structure

```
pancreas-cyst-seg/
├── .gitignore
├── README.md
├── requirements.txt
│
├── configs/
│   ├── paths.yaml              # Central path configuration (edit PANCREAS_CYST_ROOT)
│   └── __init__.py             # load_config() helper used by all scripts
│
├── data/
│   ├── prepare_dataset.py      # Raw NIfTI → nnUNet format; update split CSVs
│   ├── train.txt               # 247 image-mask pairs
│   ├── val.txt                 # 37 image-mask pairs
│   ├── test.txt                # 74 image-mask pairs
│   └── all_train.txt           # 283 pairs (train + val combined)
│
├── approach_a/                 # Direct nnUNet baseline
│   ├── train_mixed.sh          # A1: single model on full dataset
│   ├── train_per_modality.sh   # A2: one model per institution
│   ├── prepare_site_dataset.py # Helper: build per-institution nnUNet dataset
│   └── predict.sh
│
├── approach_b/                 # Cascaded: pancreas → crop → cyst
│   ├── stage1_pancreas_seg.sh  # Run PanSegNet to generate pancreas masks
│   ├── crop_to_pancreas.py     # Crop volumes to pancreas ROI + 5% margin
│   ├── paste_back.py           # Invert crop for full-space evaluation
│   ├── train.sh
│   └── predict.sh
│
├── approach_c/                 # Fine-tune PanSegNet for cyst
│   ├── finetune_trainer.py     # MONAI-based trainer with encoder warm-up
│   ├── inference.py            # Sliding-window inference
│   ├── pretrained/             # Place PanSegNet.pth here
│   ├── train.sh
│   └── predict.sh
│
├── comparison/                 # 2D U-Net and 3D V-Net baselines
│   ├── networks/               # unet.py, VNet.py, unetr.py, net_factory.py
│   ├── dataloaders/            # dataset.py (Cyst + Cyst2D), transforms
│   ├── utils/                  # losses.py, metrics.py, test_3d_patch.py
│   ├── train.py                # Unified training (--mode 2d|3d)
│   ├── test.py                 # Unified inference + metrics
│   └── evaluate.py             # Cross-approach comparison table
│
├── scripts/
│   ├── setup_gpu.sh            # One-shot env setup on a fresh GPU machine
│   ├── prepare_data.sh         # Update paths + build nnUNet raw + preprocess
│   └── set_nnunet_env.sh       # Export nnUNet_raw/preprocessed/results vars
│
└── baseline/                   # Original semi-supervised baseline code (preserved)
    ├── 2D-UNet/
    ├── 3D-VNet/
    └── environment.yaml        # Conda env for baseline models
```

---

## Installation

### Track A: nnUNet v2 Approaches (A, B, C)

Requires Python ≥ 3.10.

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 3. Install all other dependencies
pip install -r requirements.txt
```

### Track B: Comparison Baseline Models (2D U-Net, 3D V-Net)

```bash
# Requires Conda
conda env create -f baseline/environment.yaml
conda activate monai
```

---

## Data Preparation

After obtaining data access and placing images/masks in `data/images/` and
`data/masks/`:

```bash
# 1. Fix CAD mask affine headers and update split CSV paths
python data/prepare_dataset.py --config configs/paths.yaml \
    --fix-cad-headers \
    --update-txts

# 2. Build nnUNet Dataset001_PancreasCyst
python data/prepare_dataset.py --config configs/paths.yaml \
    --build-nnunet

# 3. nnUNet plan and preprocess (required for Approach A)
source scripts/set_nnunet_env.sh
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity -np 8
```

---

## Approach A

**Direct nnUNet v2 baseline** on the full mixed multi-institutional dataset.

### A1: Single Model (All Institutions Mixed)

```bash
bash approach_a/train_mixed.sh 0            # Train fold 0
bash approach_a/predict.sh "0"              # Predict test set
```

### A2: Per-Institution Models

```bash
bash approach_a/train_per_modality.sh 0    # Train one model per site
```

---

## Approach B

**Cascaded pipeline:** pretrained PanSegNet infers pancreas → crop to ROI →
train nnUNet on cropped volumes.

> **Prerequisite:** Obtain PanSegNet pretrained weights and place in
> `approach_b/pancreas_model/`. See [PanSegNet](https://github.com/mazurowski-lab/PanSegNet).

```bash
# Stage 1: Pancreas segmentation
bash approach_b/stage1_pancreas_seg.sh

# Stage 2: Crop to pancreas ROI
python approach_b/crop_to_pancreas.py --config configs/paths.yaml

# Register cropped dataset and preprocess
python data/prepare_dataset.py --config configs/paths.yaml \
    --build-nnunet --dataset-id 10
source scripts/set_nnunet_env.sh
nnUNetv2_plan_and_preprocess -d 10 --verify_dataset_integrity

# Train and predict
bash approach_b/train.sh 0
bash approach_b/predict.sh 0
```

---

## Approach C

**Fine-tune PanSegNet** encoder for binary cyst segmentation using MONAI.

> **Prerequisite:** Place `PanSegNet.pth` in `approach_c/pretrained/`.

```bash
bash approach_c/train.sh 0          # GPU 0
bash approach_c/predict.sh          # Uses best_model.pth checkpoint
```

---

## Comparison Baselines

3D V-Net and 2D U-Net with optional semi-supervised BCP training.

```bash
# 3D V-Net
python comparison/train.py --config configs/paths.yaml --mode 3d --model vnet
python comparison/test.py  --config configs/paths.yaml --mode 3d --model vnet \
    --checkpoint comparison/checkpoints/baseline/best_model.pth

# 2D U-Net
python comparison/train.py --config configs/paths.yaml --mode 2d --model unet_2d
```

---

## Evaluation

Compare all approaches on the test set:

```bash
python comparison/evaluate.py \
    --config configs/paths.yaml \
    --gt-dir data/masks \
    --split test \
    --pred-dirs \
        approach_a=approach_a/predictions/3d_fullres \
        approach_b=approach_b/predictions/full_space \
        approach_c=approach_c/predictions/test \
        vnet=comparison/predictions/vnet \
        unet2d=comparison/predictions/unet2d \
    --output results/comparison_table.csv
```

---

## Results

Results will be populated after all experiments complete.

| Approach | Dice ↑ | HD95 ↓ (mm) | ASD ↓ (mm) |
|----------|--------|-------------|------------|
| A1 — nnUNet mixed | — | — | — |
| A2 — nnUNet per-site | — | — | — |
| B — Cascaded | — | — | — |
| C — Fine-tuned PanSegNet | — | — | — |
| 3D V-Net (comparison) | — | — | — |
| 2D U-Net (comparison) | — | — | — |


