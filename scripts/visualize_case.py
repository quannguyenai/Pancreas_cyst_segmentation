#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def normalize_slice(image_2d: np.ndarray) -> np.ndarray:
    p1, p99 = np.percentile(image_2d, [1, 99])
    if p99 <= p1:
        return np.zeros_like(image_2d, dtype=np.float32)
    clipped = np.clip(image_2d, p1, p99)
    return ((clipped - p1) / (p99 - p1)).astype(np.float32)


def best_index(mask: np.ndarray, axis: int) -> int:
    sums = mask.sum(axis=tuple(i for i in range(mask.ndim) if i != axis))
    if np.all(sums == 0):
        return mask.shape[axis] // 2
    return int(np.argmax(sums))


def get_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        out = volume[index, :, :]
    elif axis == 1:
        out = volume[:, index, :]
    elif axis == 2:
        out = volume[:, :, index]
    else:
        raise ValueError(f"Unsupported axis: {axis}")
    return np.rot90(out)


def add_overlay(ax: plt.Axes, image_2d: np.ndarray, mask_2d: np.ndarray, title: str) -> None:
    base = normalize_slice(image_2d)
    ax.imshow(base, cmap="gray", interpolation="nearest")
    overlay = np.zeros((*mask_2d.shape, 4), dtype=np.float32)
    overlay[mask_2d > 0] = (1.0, 0.0, 0.0, 0.45)
    ax.imshow(overlay, interpolation="nearest")
    ax.contour(mask_2d > 0, levels=[0.5], colors=["yellow"], linewidths=0.8)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def add_mask(ax: plt.Axes, mask_2d: np.ndarray, title: str) -> None:
    ax.imshow(mask_2d > 0, cmap="gray", interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a 3D CT case and its mask.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--mask", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    image = nib.load(str(args.image)).get_fdata().astype(np.float32)
    mask = nib.load(str(args.mask)).get_fdata()
    mask = (mask > 0).astype(np.uint8)

    case_name = args.image.name.replace(".nii.gz", "")
    axes = [(2, "Axial"), (1, "Coronal"), (0, "Sagittal")]
    indices = {axis: best_index(mask, axis) for axis, _ in axes}

    fig, axs = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for col, (axis, label) in enumerate(axes):
        idx = indices[axis]
        image_2d = get_slice(image, axis, idx)
        mask_2d = get_slice(mask, axis, idx)
        add_overlay(axs[0, col], image_2d, mask_2d, f"{label} overlay (slice {idx})")
        add_mask(axs[1, col], mask_2d, f"{label} mask")

    voxel_count = int(mask.sum())
    fig.suptitle(f"{case_name} | mask voxels: {voxel_count}", fontsize=14)
    figure_path = args.out_dir / f"{case_name}_overview.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    for axis, label in axes:
        idx = indices[axis]
        image_2d = get_slice(image, axis, idx)
        mask_2d = get_slice(mask, axis, idx)
        base = normalize_slice(image_2d)
        rgb = np.stack([base, base, base], axis=-1)
        rgb[mask_2d > 0, 0] = 1.0
        rgb[mask_2d > 0, 1:] *= 0.45
        plt.imsave(args.out_dir / f"{case_name}_{label.lower()}_overlay.png", rgb)
        plt.imsave(args.out_dir / f"{case_name}_{label.lower()}_mask.png", mask_2d > 0, cmap="gray")

    summary_path = args.out_dir / f"{case_name}_summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"case={case_name}",
                f"image={args.image}",
                f"mask={args.mask}",
                f"shape={image.shape}",
                f"mask_voxels={voxel_count}",
                *(f"{label.lower()}_slice={indices[axis]}" for axis, label in axes),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
