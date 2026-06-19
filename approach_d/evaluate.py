"""approach_d/evaluate.py — Per-case Dice / HD95 / ASD on test set predictions.

Usage (from repo root):
    python approach_d/evaluate.py \
        --preds  approach_d/predictions \
        --test   data/test.txt \
        --output approach_d/test_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_nii(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).dataobj)


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    p = pred > 0
    g = gt > 0
    inter = (p & g).sum()
    denom = p.sum() + g.sum()
    return float(2 * inter / denom) if denom > 0 else 1.0


def hd95_asd(pred: np.ndarray, gt: np.ndarray):
    from medpy import metric as mpy
    p, g = pred > 0, gt > 0
    if p.sum() == 0 and g.sum() == 0:
        return 0.0, 0.0
    if p.sum() == 0 or g.sum() == 0:
        return float("nan"), float("nan")
    return float(mpy.binary.hd95(p, g)), float(mpy.binary.asd(p, g))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds",  default="approach_d/predictions")
    ap.add_argument("--test",   default="data/test.txt")
    ap.add_argument("--output", default="approach_d/test_metrics.csv")
    args = ap.parse_args()

    preds_dir = Path(args.preds)
    rows = []

    with open(args.test) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("image")]

    print(f"{'Case':<20} {'Dice':>6}  {'HD95':>7}  {'ASD':>7}")
    print("-" * 46)

    missing = []
    for line in lines:
        parts = line.split(",")
        img_path = Path(parts[0])
        gt_path  = Path(parts[1])
        stem     = img_path.name.replace(".nii.gz", "")
        pred_path = preds_dir / f"{stem}.nii.gz"

        if not pred_path.exists():
            missing.append(stem)
            continue
        if not gt_path.exists():
            print(f"[WARN] GT not found: {gt_path}")
            continue

        pred = load_nii(pred_path)
        gt   = load_nii(gt_path)

        dc = dice_score(pred, gt)
        hd, asd = hd95_asd(pred, gt)

        print(f"{stem:<20} {dc:>6.4f}  {hd:>7.2f}  {asd:>7.2f}")
        rows.append({"case": stem, "dice": dc, "hd95": hd, "asd": asd})

    if missing:
        print(f"\n[WARN] No prediction found for: {missing}")

    if not rows:
        print("No cases evaluated.")
        return

    dices = [r["dice"] for r in rows]
    hds   = [r["hd95"] for r in rows if not np.isnan(r["hd95"])]
    asds  = [r["asd"]  for r in rows if not np.isnan(r["asd"])]

    print("-" * 46)
    print(f"{'Mean':<20} {np.mean(dices):>6.4f}  {np.mean(hds):>7.2f}  {np.mean(asds):>7.2f}")
    print(f"{'Std':<20} {np.std(dices):>6.4f}  {np.std(hds):>7.2f}  {np.std(asds):>7.2f}")
    print(f"{'Median':<20} {np.median(dices):>6.4f}  {np.median(hds):>7.2f}  {np.median(asds):>7.2f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "dice", "hd95", "asd"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
