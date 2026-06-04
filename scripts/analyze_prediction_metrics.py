#!/usr/bin/env python3
"""Analyze pancreatic cyst dataset volumes and test-set prediction metrics."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd


APPROACHES = {
    "A-3D": "approach_a/prediction/3d_fullres",
    "A-2.5D": "approach_a/prediction/2d_stack5",
    "B": "approach_b/predictio/full_space",
    "D": "approach_d/prediction",
}


def read_split_file(path: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["split"] = split
    df["case"] = df["image_path"].map(lambda p: Path(p).name.replace(".nii.gz", ""))
    return df[["case", "split", "image_path", "mask_path"]]


def load_mask(path: Path) -> np.ndarray:
    return np.asanyarray(nib.load(str(path)).dataobj) > 0


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def compute_case_metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    gt = gt.astype(bool, copy=False)
    pred = pred.astype(bool, copy=False)
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: gt={gt.shape}, pred={pred.shape}")

    tp = int(np.count_nonzero(gt & pred))
    fp = int(np.count_nonzero(~gt & pred))
    fn = int(np.count_nonzero(gt & ~pred))
    tn = int(gt.size - tp - fp - fn)

    dice = safe_div(2 * tp, 2 * tp + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    accuracy = safe_div(tp + tn, gt.size)
    iou = safe_div(tp, tp + fp + fn)
    balanced_accuracy = 0.5 * (recall + specificity)
    fpr = safe_div(fp, fp + tn)
    fnr = safe_div(fn, fn + tp)
    volume_ratio = safe_div(tp + fp, tp + fn)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "gt_voxels": int(np.count_nonzero(gt)),
        "pred_voxels": int(np.count_nonzero(pred)),
        "dice": dice,
        "f1": dice,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "iou": iou,
        "fpr": fpr,
        "fnr": fnr,
        "volume_ratio_pred_gt": volume_ratio,
    }


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_cols = [
        "dice",
        "f1",
        "precision",
        "recall",
        "specificity",
        "accuracy",
        "balanced_accuracy",
        "iou",
        "fpr",
        "fnr",
        "volume_ratio_pred_gt",
    ]
    for approach, g in metrics.groupby("approach"):
        row = {"approach": approach, "n_cases": len(g)}
        for col in metric_cols:
            row[f"{col}_mean"] = g[col].mean()
            row[f"{col}_std"] = g[col].std(ddof=1)
            row[f"{col}_median"] = g[col].median()

        tp = int(g["tp"].sum())
        fp = int(g["fp"].sum())
        fn = int(g["fn"].sum())
        tn = int(g["tn"].sum())
        row.update(
            {
                "micro_tp": tp,
                "micro_fp": fp,
                "micro_fn": fn,
                "micro_tn": tn,
                "micro_dice": safe_div(2 * tp, 2 * tp + fp + fn),
                "micro_f1": safe_div(2 * tp, 2 * tp + fp + fn),
                "micro_precision": safe_div(tp, tp + fp),
                "micro_recall": safe_div(tp, tp + fn),
                "micro_specificity": safe_div(tn, tn + fp),
                "micro_accuracy": safe_div(tp + tn, tp + fp + fn + tn),
                "micro_iou": safe_div(tp, tp + fp + fn),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("dice_mean", ascending=False)


def add_volume_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["volume_bin"] = pd.cut(
        out["gt_pct"],
        bins=[0, 0.01, 0.05, 0.2, np.inf],
        labels=["<0.01%", "0.01-0.05%", "0.05-0.2%", ">=0.2%"],
        include_lowest=True,
    )
    return out


def plot_volume_distribution(volume_df: pd.DataFrame, universal: pd.DataFrame, out_path: Path) -> None:
    colors = {"train": "#4c78a8", "val": "#f58518", "test": "#54a24b"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [2.1, 1.0]})

    ax = axes[0]
    bins = np.logspace(
        np.log10(max(volume_df["gt_pct"].min() * 0.8, 1e-5)),
        np.log10(volume_df["gt_pct"].max() * 1.2),
        36,
    )
    for split, g in volume_df.groupby("split"):
        ax.hist(
            g["gt_pct"],
            bins=bins,
            alpha=0.48,
            label=f"{split} (n={len(g)})",
            color=colors.get(split),
            edgecolor="white",
            linewidth=0.4,
        )
    for i, (_, row) in enumerate(universal.sort_values("gt_pct").iterrows()):
        label = "universal failure" if i == 0 else None
        ax.axvline(row["gt_pct"], color="#d62728", linestyle="--", linewidth=1.5, alpha=0.85, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Cyst volume (% of image voxels, log scale)")
    ax.set_ylabel("Number of cases")
    ax.set_title("Cyst volume distribution by split")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1]
    order = ["train", "val", "test"]
    data = [volume_df.loc[volume_df["split"] == s, "gt_pct"].to_numpy() for s in order]
    box = ax.boxplot(data, tick_labels=order, patch_artist=True, showfliers=True)
    for patch, split in zip(box["boxes"], order):
        patch.set_facecolor(colors[split])
        patch.set_alpha(0.45)
    ax.set_yscale("log")
    ax.set_ylabel("Cyst volume (% of image voxels, log scale)")
    ax.set_title("Split balance")
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_metric_by_volume(metrics: pd.DataFrame, out_path: Path) -> None:
    df = add_volume_bins(metrics)
    summary = df.groupby(["volume_bin", "approach"], observed=True)["dice"].mean().reset_index()
    bins = list(df["volume_bin"].cat.categories)
    approaches = list(APPROACHES.keys())
    x = np.arange(len(bins))
    width = 0.18

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, approach in enumerate(approaches):
        vals = []
        for b in bins:
            hit = summary[(summary["volume_bin"] == b) & (summary["approach"] == approach)]["dice"]
            vals.append(float(hit.iloc[0]) if len(hit) else np.nan)
        ax.bar(x + (i - 1.5) * width, vals, width=width, label=approach)
    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean Dice / F1")
    ax.set_xlabel("Ground-truth cyst volume bin")
    ax.set_title("Performance by cyst size")
    ax.legend(frameon=False, ncol=4)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_dice_vs_volume(metrics: pd.DataFrame, universal: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    markers = {"A-3D": "o", "A-2.5D": "s", "B": "^", "D": "D"}
    for approach, g in metrics.groupby("approach"):
        ax.scatter(
            g["gt_pct"],
            g["dice"],
            s=32,
            alpha=0.68,
            marker=markers.get(approach, "o"),
            label=approach,
        )
    for i, (_, row) in enumerate(universal.iterrows()):
        ax.axvline(row["gt_pct"], color="#d62728", linestyle="--", linewidth=1.2, alpha=0.45)
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Cyst volume (% of image voxels, log scale)")
    ax.set_ylabel("Dice / F1")
    ax.set_title("Per-case performance vs cyst size")
    ax.legend(frameon=False, ncol=4)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_precision_recall(metrics: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for approach, g in metrics.groupby("approach"):
        ax.scatter(g["recall"], g["precision"], s=34, alpha=0.68, label=approach)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Recall / sensitivity")
    ax.set_ylabel("Precision / PPV")
    ax.set_title("Precision-recall behavior by case")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_approach_wins(metrics: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    wide = metrics.pivot(index="case", columns="approach", values="dice")
    winners = wide.idxmax(axis=1).rename("winner").reset_index()
    counts = winners["winner"].value_counts().reindex(list(APPROACHES.keys()), fill_value=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index, counts.values, color=["#4c78a8", "#72b7b2", "#f58518", "#b279a2"])
    ax.set_ylabel("Number of test cases")
    ax.set_title("Per-case Dice winner")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return winners


def write_non_a3d_comparison(metrics: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    wide = metrics.pivot(index="case", columns="approach", values="dice").reset_index()
    volume_cols = metrics[metrics["approach"] == "A-3D"][["case", "gt_voxels", "gt_pct"]]
    wide = wide.merge(volume_cols, on="case", how="left")
    non_a3d = [a for a in APPROACHES if a != "A-3D"]
    for approach in non_a3d:
        wide[f"{approach}_minus_A3D"] = wide[approach] - wide["A-3D"]
    wide["best_non_A3D"] = wide[non_a3d].idxmax(axis=1)
    wide["best_non_A3D_dice"] = wide[non_a3d].max(axis=1)
    wide["best_non_A3D_minus_A3D"] = wide["best_non_A3D_dice"] - wide["A-3D"]
    cols = [
        "case",
        "gt_voxels",
        "gt_pct",
        "A-3D",
        "A-2.5D",
        "B",
        "D",
        "best_non_A3D",
        "best_non_A3D_minus_A3D",
        "A-2.5D_minus_A3D",
        "B_minus_A3D",
        "D_minus_A3D",
    ]
    beats = wide[wide["best_non_A3D_minus_A3D"] > 0].sort_values(
        "best_non_A3D_minus_A3D", ascending=False
    )
    beats[cols].to_csv(out_dir / "cases_where_non_a3d_beats_a3d.csv", index=False)
    wide[cols].to_csv(out_dir / "per_case_delta_vs_a3d.csv", index=False)
    return wide


def plot_delta_vs_a3d(delta_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = {"A-2.5D": "#72b7b2", "B": "#f58518", "D": "#b279a2"}
    for approach in ["A-2.5D", "B", "D"]:
        ax.scatter(
            delta_df["gt_pct"],
            delta_df[f"{approach}_minus_A3D"],
            s=34,
            alpha=0.7,
            label=f"{approach} - A-3D",
            color=colors[approach],
        )
    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("Cyst volume (% of image voxels, log scale)")
    ax.set_ylabel("Dice difference vs A-3D")
    ax.set_title("Where alternatives outperform or underperform A-3D")
    ax.legend(frameon=False, ncol=3)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/huy/quan_nguyen/aima/pancrea_cyst"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--failure-threshold", type=float, default=0.2)
    args = parser.parse_args()

    root = args.root
    out_dir = args.out_dir or root / "results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = pd.concat(
        [
            read_split_file(root / "data" / "train.txt", "train"),
            read_split_file(root / "data" / "val.txt", "val"),
            read_split_file(root / "data" / "test.txt", "test"),
        ],
        ignore_index=True,
    )

    volume_rows = []
    for row in splits.itertuples(index=False):
        gt = load_mask(Path(row.mask_path))
        gt_voxels = int(np.count_nonzero(gt))
        volume_rows.append(
            {
                "case": row.case,
                "split": row.split,
                "image_path": row.image_path,
                "mask_path": row.mask_path,
                "shape": "x".join(map(str, gt.shape)),
                "image_voxels": int(gt.size),
                "gt_voxels": gt_voxels,
                "gt_pct": 100.0 * gt_voxels / gt.size,
            }
        )
    volume_df = pd.DataFrame(volume_rows)
    volume_df.to_csv(out_dir / "cyst_volume_distribution_by_split.csv", index=False)

    test_df = splits[splits["split"] == "test"].copy()
    metric_rows = []
    for row in test_df.itertuples(index=False):
        gt = load_mask(Path(row.mask_path))
        for approach, rel_dir in APPROACHES.items():
            pred_path = root / rel_dir / f"{row.case}.nii.gz"
            if not pred_path.exists():
                raise FileNotFoundError(f"Missing prediction for {approach} {row.case}: {pred_path}")
            pred = load_mask(pred_path)
            metrics = compute_case_metrics(gt, pred)
            metrics.update(
                {
                    "case": row.case,
                    "approach": approach,
                    "pred_path": str(pred_path),
                    "mask_path": row.mask_path,
                    "gt_pct": 100.0 * metrics["gt_voxels"] / gt.size,
                }
            )
            metric_rows.append(metrics)
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(out_dir / "per_case_segmentation_metrics.csv", index=False)

    summary_df = summarize_metrics(metrics_df)
    summary_df.to_csv(out_dir / "summary_segmentation_metrics.csv", index=False)

    wide = metrics_df.pivot(index="case", columns="approach", values="dice").reset_index()
    wide = wide.merge(volume_df[["case", "gt_voxels", "gt_pct"]], on="case", how="left")
    approach_cols = list(APPROACHES.keys())
    wide["max_dice"] = wide[approach_cols].max(axis=1)
    wide["mean_dice"] = wide[approach_cols].mean(axis=1)
    universal = wide[wide["max_dice"] < args.failure_threshold].sort_values(["gt_pct", "case"])
    universal.to_csv(out_dir / "universal_failure_cases.csv", index=False)

    plot_volume_distribution(
        volume_df,
        universal,
        out_dir / "cyst_volume_distribution_with_universal_failures.png",
    )
    plot_metric_by_volume(metrics_df, out_dir / "dice_by_cyst_size_bin.png")
    plot_dice_vs_volume(metrics_df, universal, out_dir / "dice_vs_cyst_volume.png")
    plot_precision_recall(metrics_df, out_dir / "precision_recall_by_case.png")
    winners = plot_approach_wins(metrics_df, out_dir / "approach_wins_by_case.png")
    winners.to_csv(out_dir / "approach_wins_by_case.csv", index=False)
    delta_df = write_non_a3d_comparison(metrics_df, out_dir)
    plot_delta_vs_a3d(delta_df, out_dir / "dice_delta_vs_a3d.png")

    by_bin = (
        add_volume_bins(metrics_df)
        .groupby(["volume_bin", "approach"], observed=True)
        .agg(
            n_cases=("case", "count"),
            dice_mean=("dice", "mean"),
            dice_median=("dice", "median"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            pred_gt_ratio_median=("volume_ratio_pred_gt", "median"),
        )
        .reset_index()
    )
    by_bin.to_csv(out_dir / "metrics_by_cyst_size_bin.csv", index=False)

    print(f"Wrote analysis outputs to: {out_dir}")
    print("Summary by approach:")
    print(summary_df[["approach", "dice_mean", "precision_mean", "recall_mean", "accuracy_mean", "iou_mean"]].to_string(index=False))
    print(f"\nUniversal failures: {len(universal)} cases with max Dice < {args.failure_threshold}")
    if len(universal):
        print(universal[["case", "gt_voxels", "gt_pct", "max_dice", "mean_dice"]].to_string(index=False))


if __name__ == "__main__":
    main()
