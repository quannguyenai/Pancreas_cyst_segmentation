#!/usr/bin/env python3
"""Visualize the worst B_priorchan cyst-segmentation cases on the cropped ROI.

Mirrors scripts/visualize_approach_b_failures.py (4-column panel:
original-before-crop / cropped ROI / cyst GT / prediction), but for the
ROI-redesign B_priorchan variant: it ranks the test cases by Dice itself and
renders the top-K worst, with GT-centered axial slices.

Usage (inside container f449bb5e955f):
  python approach_b/visualize_failures.py --config configs/paths.yaml \
      --variant priorchan --top-k 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs import load_config


# ─── helpers (mirrored from scripts/visualize_approach_b_failures.py) ──────────

def load_volume(path: Path) -> np.ndarray:
    return np.asanyarray(nib.load(str(path)).dataobj)


def normalize_slice(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32, copy=False)
    finite = np.isfinite(img)
    if not finite.any():
        return np.zeros_like(img, dtype=np.float32)
    lo, hi = np.percentile(img[finite], [1, 99])
    if hi <= lo:
        lo, hi = float(img[finite].min()), float(img[finite].max())
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - lo) / (hi - lo), 0, 1)


def nearest_resize_to_shape(mask: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    if mask.shape == target_shape:
        return mask
    coords = [
        np.clip(np.round(np.linspace(0, old - 1, new)).astype(int), 0, old - 1)
        for old, new in zip(mask.shape, target_shape)
    ]
    resized = mask
    for axis, index in enumerate(coords):
        resized = np.take(resized, index, axis=axis)
    return resized


def choose_slices(gt: np.ndarray, pred: np.ndarray, min_slices: int) -> list[int]:
    gt_counts = (gt > 0).sum(axis=(0, 1))
    pred_counts = (pred > 0).sum(axis=(0, 1))
    gt_positive = np.flatnonzero(gt_counts > 0)
    if len(gt_positive) >= min_slices:
        order = sorted(gt_positive, key=lambda z: (-int(gt_counts[z]), int(z)))
        return sorted(int(z) for z in order[:min_slices])
    selected = [int(z) for z in gt_positive]
    if selected:
        center = int(round(float(np.mean(selected))))
    elif pred_counts.any():
        center = int(np.argmax(pred_counts))
    else:
        center = gt.shape[2] // 2
    radius = 0
    while len(selected) < min_slices and len(selected) < gt.shape[2]:
        for z in (center - radius, center + radius):
            if 0 <= z < gt.shape[2] and z not in selected:
                selected.append(int(z))
                if len(selected) >= min_slices:
                    break
        radius += 1
    return sorted(selected)


def show_base(ax: plt.Axes, image_slice: np.ndarray) -> None:
    ax.imshow(np.rot90(normalize_slice(image_slice)), cmap="gray", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])


def overlay_mask(ax: plt.Axes, mask_slice: np.ndarray, color: str) -> None:
    mask = np.rot90(mask_slice > 0)
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    colors = {"gt": (1.0, 0.15, 0.05, 0.55), "pred": (0.0, 0.75, 1.0, 0.55)}
    rgba[mask] = colors[color]
    ax.imshow(rgba, interpolation="nearest")


def dice(gt: np.ndarray, pred: np.ndarray) -> float:
    gt = gt.astype(bool); pred = pred.astype(bool)
    denom = int(gt.sum() + pred.sum())
    return 1.0 if denom == 0 else 2 * int((gt & pred).sum()) / denom


# ─── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", default="configs/paths.yaml")
    p.add_argument("--variant", default="priorchan", help="Variant prediction subdir under approach_b/predictions/.")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--slices-per-case", type=int, default=5)
    p.add_argument("--out", default=None, help="Output PNG (default results/analysis/B_<variant>_top_failed.png).")
    args = p.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["root"])
    original_images = Path(cfg["data"]["images"])
    cropped_images = Path(cfg["approach_b"]["cropped_images"])   # roi256/images
    cropped_masks = Path(cfg["approach_b"]["cropped_masks"])     # roi256/masks
    cropped_preds = root / "approach_b" / "predictions" / args.variant / "cropped"
    crop_stats = json.loads((root / "approach_b" / "crop_stats.json").read_text())

    out_path = Path(args.out) if args.out else root / "results" / "analysis" / f"B_{args.variant}_top_failed.png"

    # Rank test cases by Dice on the cropped grid.
    test_df = pd.read_csv(cfg["data"]["test_txt"])
    test_df["case"] = test_df["image_path"].map(lambda x: Path(x).name.replace(".nii.gz", ""))

    metrics = []
    for case in test_df["case"]:
        img_p = cropped_images / f"{case}.nii.gz"
        gt_p = cropped_masks / f"{case}.nii.gz"
        pr_p = cropped_preds / f"{case}.nii.gz"
        if not (img_p.exists() and gt_p.exists() and pr_p.exists() and case in crop_stats):
            continue
        img = load_volume(img_p)
        gt = nearest_resize_to_shape(load_volume(gt_p) > 0, img.shape)
        pred = nearest_resize_to_shape(load_volume(pr_p) > 0, img.shape)
        metrics.append({
            "case": case, "dice": dice(gt, pred),
            "gt_voxels": int(gt.sum()), "pred_voxels": int(pred.sum()),
        })
    mdf = pd.DataFrame(metrics).sort_values("dice").reset_index(drop=True)
    top = mdf.head(args.top_k)
    print(f"B_{args.variant}: {len(mdf)} test cases scored; worst {len(top)}:")
    print(top.to_string(index=False))

    selected = []
    for _, row in top.iterrows():
        case = row["case"]
        original = load_volume(original_images / f"{case}.nii.gz")
        image = load_volume(cropped_images / f"{case}.nii.gz")
        gt = nearest_resize_to_shape(load_volume(cropped_masks / f"{case}.nii.gz") > 0, image.shape)
        pred = nearest_resize_to_shape(load_volume(cropped_preds / f"{case}.nii.gz") > 0, image.shape)
        slices = choose_slices(gt, pred, args.slices_per_case)
        selected.append((row, original, image, gt, pred, crop_stats[case], slices))

    if not selected:
        raise RuntimeError("No cases could be visualized; check prediction/crop paths.")

    n_rows = sum(len(item[-1]) for item in selected)
    fig, axes = plt.subplots(n_rows, 4, figsize=(13.2, 1.95 * n_rows), squeeze=False)
    for ax, title in zip(axes[0], ["Original image before crop", "Cropped pancreas ROI",
                                   "Cyst ground truth", f"B_{args.variant} prediction"]):
        ax.set_title(title, fontsize=12, pad=8)

    plot_row = 0
    for row, original, image, gt, pred, stats, slices in selected:
        bbox_start = stats["bbox_start"]
        for slice_idx, z in enumerate(slices):
            original_z = int(np.clip(int(bbox_start[2]) + z, 0, original.shape[2] - 1))
            show_base(axes[plot_row, 0], original[:, :, original_z])
            for col in range(1, 4):
                show_base(axes[plot_row, col], image[:, :, z])
            overlay_mask(axes[plot_row, 2], gt[:, :, z], "gt")
            overlay_mask(axes[plot_row, 3], pred[:, :, z], "pred")
            label = f"{row['case']} | Dice {row['dice']:.3f}" if slice_idx == 0 else ""
            axes[plot_row, 0].set_ylabel(label, fontsize=9, rotation=0, ha="right", va="center")
            axes[plot_row, 1].set_ylabel(f"crop z={z}\norig z={original_z}", fontsize=7)
            plot_row += 1

    fig.suptitle(f"B_{args.variant} top {len(selected)} failed cases "
                 f"({args.slices_per_case} GT-centered slices/case) | red=GT, blue=pred",
                 fontsize=14, y=0.995)
    fig.tight_layout(rect=(0.095, 0, 1, 0.985))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    top.to_csv(out_path.with_suffix(".csv"), index=False)
    print(f"\nWrote figure:   {out_path}")
    print(f"Wrote case list: {out_path.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
