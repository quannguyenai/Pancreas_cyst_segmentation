#!/usr/bin/env python3
"""EDA for the pancrea cyst NIfTI dataset."""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd


DATA_DIR = Path("/home/huy/quan_nguyen/aima/pancrea_cyst/data")
OUT_DIR = Path("/home/huy/quan_nguyen/aima/pancrea_cyst/eda")


def patient_id_from_image(path: str | Path) -> str:
    return Path(path).name.removesuffix(".nii.gz")


def patient_id_from_mask(path: str | Path) -> str:
    name = Path(path).name.removesuffix(".nii.gz")
    return re.sub(r"^cyst_", "", name).replace("_", "").upper()


def site_from_case(case_id: str) -> str:
    match = re.match(r"([A-Z]+)", case_id)
    return match.group(1) if match else "UNKNOWN"


def read_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["image_path", "mask_path", "case_id", "site"])
    df = pd.read_csv(path)
    df["case_id"] = df["image_path"].map(patient_id_from_image)
    df["mask_case_id"] = df["mask_path"].map(patient_id_from_mask)
    df["site"] = df["case_id"].map(site_from_case)
    return df


def describe_numeric(series: pd.Series) -> dict[str, float]:
    return {
        "min": float(series.min()),
        "p25": float(series.quantile(0.25)),
        "median": float(series.median()),
        "mean": float(series.mean()),
        "p75": float(series.quantile(0.75)),
        "max": float(series.max()),
    }


def save_bar(counter: pd.Series, title: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    counter.plot(kind="bar", ax=ax, color="#4062bb")
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_hist(series: pd.Series, title: str, xlabel: str, path: Path, bins: int = 32) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(series.dropna(), bins=bins, color="#2f9c95", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cases")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_representative_montage(cases: pd.DataFrame, path: Path) -> None:
    non_empty = cases[cases["mask_voxels"] > 0].sort_values("mask_volume_ml")
    if non_empty.empty:
        return

    picks = [
        ("small", non_empty.iloc[0]),
        ("median", non_empty.iloc[len(non_empty) // 2]),
        ("large", non_empty.iloc[-1]),
    ]
    fig, axes = plt.subplots(1, len(picks), figsize=(12, 4))
    for ax, (label, row) in zip(axes, picks):
        image = nib.load(row["image_path"])
        mask = nib.load(row["mask_path"])
        img_data = np.asanyarray(image.dataobj)
        mask_data = np.asanyarray(mask.dataobj) > 0

        z_indices = np.where(mask_data.any(axis=(0, 1)))[0]
        z = int(z_indices[len(z_indices) // 2]) if len(z_indices) else img_data.shape[2] // 2
        img_slice = np.rot90(img_data[:, :, z])
        mask_slice = np.rot90(mask_data[:, :, z])

        lo, hi = np.percentile(img_slice[np.isfinite(img_slice)], [1, 99])
        ax.imshow(img_slice, cmap="gray", vmin=lo, vmax=hi)
        ax.imshow(np.ma.masked_where(~mask_slice, mask_slice), cmap="autumn", alpha=0.55)
        ax.set_title(f"{label}: {row['case_id']}\n{row['mask_volume_ml']:.2f} mL")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted((DATA_DIR / "images").glob("*.nii.gz"))
    mask_paths = sorted((DATA_DIR / "masks").glob("*.nii.gz"))
    image_ids = {patient_id_from_image(p): p for p in image_paths}
    mask_ids = {patient_id_from_mask(p): p for p in mask_paths}

    split_frames = {}
    for split in ["train", "val", "test", "all_train"]:
        split_frames[split] = read_split(DATA_DIR / f"{split}.txt")

    rows = []
    mask_value_counter: Counter[str] = Counter()
    for idx, case_id in enumerate(sorted(image_ids)):
        image_path = image_ids[case_id]
        mask_path = mask_ids.get(case_id)

        image = nib.load(str(image_path))
        shape = tuple(int(v) for v in image.shape[:3])
        spacing = tuple(float(v) for v in image.header.get_zooms()[:3])
        voxel_volume_mm3 = float(np.prod(spacing))

        img_data = np.asanyarray(image.dataobj)
        finite = np.isfinite(img_data)
        finite_data = img_data[finite]

        mask_voxels = np.nan
        mask_volume_ml = np.nan
        bbox_dims_vox = (np.nan, np.nan, np.nan)
        bbox_dims_mm = (np.nan, np.nan, np.nan)
        mask_values = ""
        mask_shape_match = False
        affine_match = False
        empty_mask = np.nan

        if mask_path is not None:
            mask = nib.load(str(mask_path))
            mask_shape_match = tuple(mask.shape[:3]) == shape
            affine_match = bool(np.allclose(image.affine, mask.affine, atol=1e-3))
            mask_data = np.asanyarray(mask.dataobj)
            unique_values = np.unique(mask_data)
            mask_values = "|".join(str(float(v)).rstrip("0").rstrip(".") for v in unique_values[:20])
            for value in unique_values:
                mask_value_counter[str(float(value)).rstrip("0").rstrip(".")] += 1

            positive = mask_data > 0
            mask_voxels = int(positive.sum())
            empty_mask = mask_voxels == 0
            mask_volume_ml = mask_voxels * voxel_volume_mm3 / 1000.0
            if mask_voxels:
                coords = np.argwhere(positive)
                mins = coords.min(axis=0)
                maxs = coords.max(axis=0)
                bbox_vox = maxs - mins + 1
                bbox_dims_vox = tuple(int(v) for v in bbox_vox)
                bbox_dims_mm = tuple(float(v) for v in bbox_vox * np.array(spacing))

        rows.append(
            {
                "case_id": case_id,
                "site": site_from_case(case_id),
                "image_path": str(image_path),
                "mask_path": str(mask_path) if mask_path is not None else "",
                "shape_x": shape[0],
                "shape_y": shape[1],
                "shape_z": shape[2],
                "spacing_x_mm": spacing[0],
                "spacing_y_mm": spacing[1],
                "spacing_z_mm": spacing[2],
                "voxel_volume_mm3": voxel_volume_mm3,
                "image_min": float(finite_data.min()) if finite_data.size else np.nan,
                "image_p01": float(np.percentile(finite_data, 1)) if finite_data.size else np.nan,
                "image_median": float(np.median(finite_data)) if finite_data.size else np.nan,
                "image_p99": float(np.percentile(finite_data, 99)) if finite_data.size else np.nan,
                "image_max": float(finite_data.max()) if finite_data.size else np.nan,
                "mask_voxels": mask_voxels,
                "mask_volume_ml": mask_volume_ml,
                "mask_fraction": mask_voxels / np.prod(shape) if mask_path is not None else np.nan,
                "bbox_x_vox": bbox_dims_vox[0],
                "bbox_y_vox": bbox_dims_vox[1],
                "bbox_z_vox": bbox_dims_vox[2],
                "bbox_x_mm": bbox_dims_mm[0],
                "bbox_y_mm": bbox_dims_mm[1],
                "bbox_z_mm": bbox_dims_mm[2],
                "mask_values": mask_values,
                "mask_shape_match": mask_shape_match,
                "affine_match": affine_match,
                "empty_mask": empty_mask,
            }
        )

        if (idx + 1) % 50 == 0:
            print(f"processed {idx + 1}/{len(image_ids)}")

    cases = pd.DataFrame(rows)
    for split_name, split_df in split_frames.items():
        cases[f"in_{split_name}"] = cases["case_id"].isin(set(split_df["case_id"]))

    cases.to_csv(OUT_DIR / "case_level_eda.csv", index=False)

    split_summary_rows = []
    for split_name, split_df in split_frames.items():
        split_summary_rows.append(
            {
                "split": split_name,
                "rows": len(split_df),
                "unique_cases": split_df["case_id"].nunique(),
                "mismatched_image_mask_ids": int((split_df["case_id"] != split_df["mask_case_id"]).sum())
                if len(split_df)
                else 0,
                "missing_image_files": int((~split_df["image_path"].map(lambda p: Path(p).exists())).sum())
                if len(split_df)
                else 0,
                "missing_mask_files": int((~split_df["mask_path"].map(lambda p: Path(p).exists())).sum())
                if len(split_df)
                else 0,
            }
        )
    split_summary = pd.DataFrame(split_summary_rows)
    split_summary.to_csv(OUT_DIR / "split_summary.csv", index=False)

    site_by_split = []
    for split_name, split_df in split_frames.items():
        for site, count in split_df["site"].value_counts().sort_index().items():
            site_by_split.append({"split": split_name, "site": site, "cases": int(count)})
    pd.DataFrame(site_by_split).to_csv(OUT_DIR / "site_by_split.csv", index=False)

    save_bar(cases["site"].value_counts().sort_index(), "Cases by Site", "Cases", OUT_DIR / "cases_by_site.png")
    save_hist(cases["mask_volume_ml"], "Cyst Mask Volume Distribution", "Mask volume (mL)", OUT_DIR / "mask_volume_hist.png")
    save_hist(np.log10(cases["mask_volume_ml"].replace(0, np.nan)), "Cyst Mask Volume Distribution (log10)", "log10(mask volume mL)", OUT_DIR / "mask_volume_log_hist.png")
    save_representative_montage(cases, OUT_DIR / "representative_mask_overlay.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(cases["shape_z"], cases["spacing_z_mm"], s=30, alpha=0.75, color="#b24c63")
    ax.set_title("Slice Count vs Slice Thickness")
    ax.set_xlabel("Z slices")
    ax.set_ylabel("Z spacing (mm)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "slice_count_vs_spacing.png", dpi=160)
    plt.close(fig)

    volume_stats = describe_numeric(cases["mask_volume_ml"])
    z_spacing_stats = describe_numeric(cases["spacing_z_mm"])
    voxel_volume_stats = describe_numeric(cases["voxel_volume_mm3"])

    top_large = cases.nlargest(10, "mask_volume_ml")[
        ["case_id", "site", "shape_x", "shape_y", "shape_z", "spacing_x_mm", "spacing_y_mm", "spacing_z_mm", "mask_volume_ml"]
    ]
    top_small = cases[cases["mask_voxels"] > 0].nsmallest(10, "mask_volume_ml")[
        ["case_id", "site", "shape_x", "shape_y", "shape_z", "spacing_x_mm", "spacing_y_mm", "spacing_z_mm", "mask_volume_ml"]
    ]

    with (OUT_DIR / "eda_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Pancrea Cyst Dataset EDA\n\n")
        f.write(f"- Data directory: `{DATA_DIR}`\n")
        f.write(f"- Image files: {len(image_paths)}\n")
        f.write(f"- Mask files: {len(mask_paths)}\n")
        f.write(f"- Paired cases by ID: {len(set(image_ids) & set(mask_ids))}\n")
        f.write(f"- Images without masks: {len(set(image_ids) - set(mask_ids))}\n")
        f.write(f"- Masks without images: {len(set(mask_ids) - set(image_ids))}\n")
        f.write(f"- Empty positive masks: {int(cases['empty_mask'].fillna(False).sum())}\n")
        f.write(f"- Shape mismatches: {int((~cases['mask_shape_match']).sum())}\n")
        f.write(f"- Affine mismatches: {int((~cases['affine_match']).sum())}\n")
        f.write(f"- Observed mask values: {dict(sorted(mask_value_counter.items()))}\n\n")

        f.write("## Splits\n\n")
        f.write(split_summary.to_markdown(index=False))
        f.write("\n\n")
        pivot = pd.DataFrame(site_by_split).pivot_table(index="site", columns="split", values="cases", fill_value=0, aggfunc="sum")
        f.write(pivot.to_markdown())
        f.write("\n\n")

        f.write("## Geometry\n\n")
        f.write(f"- Unique image shapes: {cases[['shape_x', 'shape_y', 'shape_z']].drop_duplicates().shape[0]}\n")
        f.write(f"- Unique voxel spacings: {cases[['spacing_x_mm', 'spacing_y_mm', 'spacing_z_mm']].drop_duplicates().shape[0]}\n")
        f.write(f"- Z spacing stats (mm): {z_spacing_stats}\n")
        f.write(f"- Voxel volume stats (mm^3): {voxel_volume_stats}\n\n")

        f.write("## Mask Volumes\n\n")
        f.write(f"- Mask volume stats (mL): {volume_stats}\n")
        f.write(f"- Median mask fraction of image volume: {float(cases['mask_fraction'].median()):.8f}\n\n")
        f.write("### 10 Largest Masks\n\n")
        f.write(top_large.to_markdown(index=False))
        f.write("\n\n### 10 Smallest Non-empty Masks\n\n")
        f.write(top_small.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Output Files\n\n")
        for output in sorted(OUT_DIR.iterdir()):
            if output.name != "eda_summary.md":
                f.write(f"- `{output.name}`\n")

    print(f"Wrote EDA outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
