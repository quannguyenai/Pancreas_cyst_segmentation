#!/usr/bin/env python3
"""Visualize the worst Approach B cyst-segmentation cases on cropped ROIs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a panel of top failed Approach B cases.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--slices-per-case",
        type=int,
        default=5,
        help="Minimum number of axial slices to show for each case.",
    )
    parser.add_argument(
        "--metrics",
        default="results/per_case/approach_b.csv",
        help="Approach B per-case metrics CSV, relative to repo root unless absolute.",
    )
    parser.add_argument(
        "--out",
        default="results/analysis/approach_b_top_failed_cropped_pancreas.png",
        help="Output PNG path, relative to repo root unless absolute.",
    )
    parser.add_argument(
        "--out-csv",
        default="results/analysis/approach_b_top_failed_cases.csv",
        help="Output CSV of selected cases, relative to repo root unless absolute.",
    )
    return parser.parse_args()


def resolve_path(repo_root: Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else repo_root / path


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
    """Resize a label mask with nearest-neighbor indexing without extra deps."""
    if mask.shape == target_shape:
        return mask
    coords = [
        np.clip(
            np.round(np.linspace(0, old - 1, new)).astype(int),
            0,
            old - 1,
        )
        for old, new in zip(mask.shape, target_shape)
    ]
    resized = mask
    for axis, index in enumerate(coords):
        resized = np.take(resized, index, axis=axis)
    return resized


def choose_slice(gt: np.ndarray, pred: np.ndarray) -> int:
    union = (gt > 0) | (pred > 0)
    if union.any():
        return int(np.argmax(union.sum(axis=(0, 1))))
    return gt.shape[2] // 2


def choose_slices(gt: np.ndarray, pred: np.ndarray, min_slices: int) -> list[int]:
    """Pick a compact set of slices, prioritizing slices with GT cyst voxels."""
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
    colors = {
        "gt": (1.0, 0.15, 0.05, 0.55),
        "pred": (0.0, 0.75, 1.0, 0.55),
    }
    rgba[mask] = colors[color]
    ax.imshow(rgba, interpolation="nearest")


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    metrics_path = resolve_path(repo_root, args.metrics)
    out_path = resolve_path(repo_root, args.out)
    out_csv = resolve_path(repo_root, args.out_csv)

    original_images = repo_root / "data" / "images"
    cropped_images = repo_root / "approach_b" / "cropped" / "images"
    cropped_masks = repo_root / "approach_b" / "cropped" / "masks"
    cropped_preds = repo_root / "approach_b" / "predictio" / "cropped"
    crop_stats_path = repo_root / "approach_b" / "crop_stats.json"
    crop_stats = json.loads(crop_stats_path.read_text())

    metrics = pd.read_csv(metrics_path)
    top = (
        metrics.sort_values(["dice", "asd", "hd95"], ascending=[True, False, False])
        .head(args.top_k)
        .copy()
    )

    rows: list[dict[str, object]] = []
    selected: list[
        tuple[pd.Series, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object], list[int]]
    ] = []
    for _, row in top.iterrows():
        case = str(row["case"])
        original_path = original_images / f"{case}.nii.gz"
        image_path = cropped_images / f"{case}.nii.gz"
        gt_path = cropped_masks / f"{case}.nii.gz"
        pred_path = cropped_preds / f"{case}.nii.gz"
        missing = [str(p) for p in (original_path, image_path, gt_path, pred_path) if not p.exists()]
        if case not in crop_stats:
            missing.append(f"{crop_stats_path}:{case}")
        if missing:
            print(f"[WARN] Skipping {case}; missing: {', '.join(missing)}")
            continue

        original = load_volume(original_path)
        image = load_volume(image_path)
        gt = load_volume(gt_path) > 0
        pred = load_volume(pred_path) > 0
        gt = nearest_resize_to_shape(gt, image.shape)
        pred = nearest_resize_to_shape(pred, image.shape)
        slices = choose_slices(gt, pred, args.slices_per_case)
        selected.append((row, original, image, gt, pred, crop_stats[case], slices))
        bbox_start = crop_stats[case]["bbox_start"]
        rows.append(
            {
                "case": case,
                "dice": float(row["dice"]),
                "hd95": float(row["hd95"]),
                "asd": float(row["asd"]),
                "displayed_slices_cropped_z": " ".join(str(z) for z in slices),
                "displayed_slices_original_z": " ".join(str(int(bbox_start[2]) + z) for z in slices),
                "gt_voxels_cropped": int(np.count_nonzero(gt)),
                "pred_voxels_cropped": int(np.count_nonzero(pred)),
                "original_image": str(original_path),
                "cropped_image": str(image_path),
                "cropped_groundtruth": str(gt_path),
                "cropped_prediction": str(pred_path),
            }
        )

    if not selected:
        raise RuntimeError("No cases could be visualized; check the cropped data paths.")

    n_cases = len(selected)
    n_rows = sum(len(item[-1]) for item in selected)
    fig, axes = plt.subplots(n_rows, 4, figsize=(13.2, 1.95 * n_rows), squeeze=False)
    column_titles = [
        "Original image before crop",
        "Cropped pancreas ROI",
        "Cyst ground truth",
        "Approach B prediction",
    ]
    for ax, title in zip(axes[0], column_titles):
        ax.set_title(title, fontsize=12, pad=8)

    plot_row = 0
    slice_rows: list[dict[str, object]] = []
    for row, original, image, gt, pred, stats, slices in selected:
        bbox_start = stats["bbox_start"]
        for slice_idx, z in enumerate(slices):
            original_z = int(bbox_start[2]) + z
            original_z = int(np.clip(original_z, 0, original.shape[2] - 1))

            original_slice = original[:, :, original_z]
            image_slice = image[:, :, z]
            gt_slice = gt[:, :, z]
            pred_slice = pred[:, :, z]

            show_base(axes[plot_row, 0], original_slice)
            for col in range(1, 4):
                show_base(axes[plot_row, col], image_slice)
            overlay_mask(axes[plot_row, 2], gt_slice, "gt")
            overlay_mask(axes[plot_row, 3], pred_slice, "pred")

            if slice_idx == 0:
                label = f"{row['case']} | Dice {row['dice']:.3f} | ASD {row['asd']:.1f}"
            else:
                label = ""
            axes[plot_row, 0].set_ylabel(label, fontsize=9, rotation=0, ha="right", va="center")
            axes[plot_row, 1].set_ylabel(f"crop z={z}\norig z={original_z}", fontsize=7)

            slice_rows.append(
                {
                    "case": row["case"],
                    "dice": float(row["dice"]),
                    "hd95": float(row["hd95"]),
                    "asd": float(row["asd"]),
                    "cropped_z": z,
                    "original_z": original_z,
                    "gt_voxels_on_slice": int(np.count_nonzero(gt_slice)),
                    "pred_voxels_on_slice": int(np.count_nonzero(pred_slice)),
                }
            )
            plot_row += 1

    fig.suptitle(
        f"Approach B top {n_cases} failed cases, {args.slices_per_case} GT-centered slices per case",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=(0.095, 0, 1, 0.985))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv = out_csv
    slice_csv = out_csv.with_name(out_csv.stem + "_slices.csv")
    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    pd.DataFrame(slice_rows).to_csv(slice_csv, index=False)
    print(f"Wrote figure: {out_path}")
    print(f"Wrote case list: {summary_csv}")
    print(f"Wrote slice list: {slice_csv}")


if __name__ == "__main__":
    main()
