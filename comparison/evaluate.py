"""evaluate.py — Cross-approach evaluation and comparison table.

Computes Dice, HD95, and ASD for each approach's predictions against
ground-truth masks, then prints and saves a comparison table.

Usage
-----
python comparison/evaluate.py \\
    --config configs/paths.yaml \\
    --gt-dir data/masks \\
    --pred-dirs approach_a=approach_a/predictions/3d_fullres \\
                approach_b=approach_b/predictions/full_space \\
                approach_c=approach_c/predictions \\
                vnet=comparison/predictions/vnet \\
                unet2d=comparison/predictions/unet2d \\
    --output results/comparison_table.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs import load_config
from comparison.utils.metrics import calculate_metric_percase


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate and compare predictions across approaches.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="configs/paths.yaml")
    p.add_argument(
        "--gt-dir", required=True,
        help="Directory containing ground-truth NIfTI masks (data/masks/).",
    )
    p.add_argument(
        "--pred-dirs", nargs="+", required=True,
        metavar="NAME=PATH",
        help="One or more name=path pairs of prediction directories.",
    )
    p.add_argument(
        "--output", default="results/comparison_table.csv",
        help="Where to save the CSV comparison table.",
    )
    p.add_argument(
        "--split", default="test", choices=["train", "val", "test"],
        help="Which split to evaluate on (used to restrict case list).",
    )
    return p.parse_args()


def load_cases_from_split(cfg: dict, split: str) -> list[str]:
    """Return list of image stems from a split CSV."""
    key = {"train": "train_txt", "val": "val_txt", "test": "test_txt"}[split]
    txt_path = Path(cfg["data"][key])
    stems: list[str] = []
    with open(txt_path) as f:
        for line in f.readlines()[1:]:
            line = line.strip()
            if line:
                img_path = line.split(",")[0]
                stems.append(Path(img_path).name.replace(".nii.gz", ""))
    return stems


def evaluate_folder(
    pred_dir: Path,
    gt_dir: Path,
    case_stems: list[str],
) -> pd.DataFrame:
    """Compute per-case metrics for all predictions in pred_dir.

    Returns a DataFrame with columns: case, dice, hd95, asd.
    """
    rows: list[dict] = []
    missing = 0

    for stem in case_stems:
        pred_path = pred_dir / f"{stem}.nii.gz"
        # Masks use cyst_ prefix naming convention
        # Try both formats
        gt_candidates = list(gt_dir.glob(f"*{stem.lower()}*.nii.gz"))
        if not gt_candidates:
            # Try direct match
            gt_path = gt_dir / f"{stem}.nii.gz"
        else:
            gt_path = gt_candidates[0]

        if not pred_path.exists():
            missing += 1
            continue
        if not gt_path.exists():
            continue

        pred = nib.load(str(pred_path)).get_fdata().astype(np.uint8)
        gt   = nib.load(str(gt_path)).get_fdata().astype(np.uint8)

        if gt.sum() == 0:
            continue  # skip cases with no GT annotation

        try:
            dice, hd95, asd, _ = calculate_metric_percase(pred > 0, gt > 0)
        except Exception:
            dice, hd95, asd = 0.0, float("nan"), float("nan")

        rows.append({"case": stem, "dice": dice, "hd95": hd95, "asd": asd})

    if missing:
        print(f"  [WARN] {missing} predictions missing in {pred_dir.name}")

    return pd.DataFrame(rows)


def compare_approaches(
    approach_dirs: dict[str, Path],
    gt_dir: Path,
    case_stems: list[str],
    output_csv: Path,
) -> None:
    """Evaluate all approaches and print/save a comparison table."""
    summary_rows: list[dict] = []
    per_case_frames: dict[str, pd.DataFrame] = {}

    for name, pred_dir in approach_dirs.items():
        if not pred_dir.exists():
            print(f"[SKIP] {name}: directory not found ({pred_dir})")
            continue

        print(f"Evaluating {name} ...")
        df = evaluate_folder(pred_dir, gt_dir, case_stems)
        per_case_frames[name] = df

        if df.empty:
            print(f"  [WARN] No valid predictions found for {name}")
            continue

        summary_rows.append({
            "approach": name,
            "n_cases": len(df),
            "dice_mean": df["dice"].mean(),
            "dice_std":  df["dice"].std(),
            "hd95_mean": df["hd95"].mean(),
            "hd95_std":  df["hd95"].std(),
            "asd_mean":  df["asd"].mean(),
            "asd_std":   df["asd"].std(),
        })

    if not summary_rows:
        print("No results to report.")
        return

    summary = pd.DataFrame(summary_rows)

    # Print formatted table
    print(f"\n{'='*80}")
    print(f"{'Approach':<20} {'N':>5} {'Dice':>10} {'HD95':>10} {'ASD':>10}")
    print(f"{'-'*20} {'-'*5} {'-'*10} {'-'*10} {'-'*10}")
    for _, row in summary.iterrows():
        print(
            f"{row['approach']:<20} {row['n_cases']:>5} "
            f"{row['dice_mean']:>6.4f}±{row['dice_std']:.4f} "
            f"{row['hd95_mean']:>6.2f}±{row['hd95_std']:.2f} "
            f"{row['asd_mean']:>6.2f}±{row['asd_std']:.2f}"
        )
    print(f"{'='*80}\n")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False, float_format="%.4f")
    print(f"Summary saved to: {output_csv}")

    # Also save per-case CSVs
    per_case_dir = output_csv.parent / "per_case"
    per_case_dir.mkdir(exist_ok=True)
    for name, df in per_case_frames.items():
        df.to_csv(per_case_dir / f"{name}.csv", index=False, float_format="%.4f")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    gt_dir = Path(args.gt_dir)
    case_stems = load_cases_from_split(cfg, args.split)
    print(f"Evaluating {len(case_stems)} cases from {args.split} split.")

    approach_dirs: dict[str, Path] = {}
    for entry in args.pred_dirs:
        if "=" not in entry:
            print(f"[ERROR] --pred-dirs entries must be NAME=PATH, got: {entry!r}")
            sys.exit(1)
        name, path = entry.split("=", 1)
        approach_dirs[name] = Path(path)

    compare_approaches(
        approach_dirs=approach_dirs,
        gt_dir=gt_dir,
        case_stems=case_stems,
        output_csv=Path(args.output),
    )


if __name__ == "__main__":
    main()
