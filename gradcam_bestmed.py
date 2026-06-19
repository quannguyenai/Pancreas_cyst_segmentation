"""pancrea_cyst/gradcam_bestmed.py — Seg-Grad-CAM fail/best/medium figures for the MRI A-3D model.

A-3D = nnU-Net v2 PlainConvUNet (Dataset001_PancreasCyst_3DA, 3d_fullres). Uses the shared
seg_comparison/gradcam_lib with this model's plans.json — CTNormalization (clip 55/5491,
mean 555, std 665) faithfully reproduces the deployed model (matches results/figures/gradcam_ct/).

Emits the 4-column figure [MRI | GT | Grad-CAM+Pred | MRI+CAM+GT+Pred], categories fail/best/medium.
Run with the CT-cyst venv (has the nnU-Net v2 stack):
  CUDA_VISIBLE_DEVICES=0 /raid/team/team/pancreas_ct_cyst_seg/.venv/bin/python gradcam_bestmed.py
Out: results/figures/gradcam_a3d/gradcam_<cat>_<case>.png
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, "/raid/team/team/seg_comparison")
import gradcam_lib as G

REPO = Path("/raid/team/team/pancrea_cyst")
CKPT = REPO / "nnUnet/nnUNet_results/Dataset001_PancreasCyst_3DA/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"
PLANS = REPO / "nnUnet/nnUNet_results/Dataset001_PancreasCyst_3DA/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json"
IMG = REPO / "data/images"
MASK = REPO / "data/masks"
PRED = REPO / "approach_a/prediction/3d_fullres"
FIG = REPO / "results/figures/gradcam_a3d"


def mask_stem(stem: str) -> str:
    m = re.match(r"^([A-Za-z]+?)(\d+)$", stem)
    return f"cyst_{m.group(1).lower()}_{m.group(2)}"


device = "cuda:0" if torch.cuda.is_available() else "cpu"
cfg = G.load_plans_cfg(PLANS)
print(f"Loading MRI A-3D model (PlainConvUNet) on {device} ...")
net = G.load_model(PLANS, CKPT).to(device)
print("Model loaded.")

df = pd.read_csv(REPO / "results/comparison_by_case.csv")
df["case"] = df["case"].astype(str)

# fail = bottom-5 Dice (the universal misses); best = top-5; medium = band 0.30-0.65
fail = [str(c) for c in df.nsmallest(5, "A_3D_dice")["case"].tolist()]
best, medium = G.select_best_medium(df, dice_col="A_3D_dice", n=5, exclude=fail)
CASES = {"fail": fail, "best": best, "medium": medium}
print("fail:", fail, "\nbest:", best, "\nmedium:", medium)

df = df.set_index("case")
for cat, case_list in CASES.items():
    print(f"--- {cat} ---")
    for case in case_list:
        r = df.loc[case]
        subtitle = (f"[{cat}] MRI A-3D  Dice={r['A_3D_dice']:.3f}  (A2.5D={r['A_25D_dice']:.2f} "
                    f"B={r['B_dice']:.2f} D={r['D_dice']:.2f})  site={r['site']}")
        G.make_gradcam_figure(
            case=case,
            img_path=IMG / f"{case}.nii.gz",
            gt_path=MASK / f"{mask_stem(case)}.nii.gz",
            pred_path=PRED / f"{case}.nii.gz",
            net=net, cfg=cfg, out_path=FIG / f"gradcam_{cat}_{case}.png",
            modality="MRI", subtitle=subtitle, device=device,
        )

print(f"\nDone. Figures in {FIG}")
