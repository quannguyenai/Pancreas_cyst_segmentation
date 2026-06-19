"""eval_variants.py — Full-space test-set Dice for Approach-B ROI variants.

Compares each variant's pasted-back predictions against the GT cyst masks listed
in the test split, using binary Dice (empty-gt & empty-pred => 1.0). Reports
mean/median/std plus catastrophic-failure counts.

Usage:
  python approach_b/eval_variants.py --config configs/paths.yaml
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

# variant name -> full-space prediction dir (relative to repo root)
VARIANT_DIRS = {
    "B_baseline":  "approach_b/predictions/full_space",          # D010 tight+15%
    "B_fixedbox":  "approach_b/predictions/fixedbox/full_space",  # D013
    "B_priorchan": "approach_b/predictions/priorchan/full_space", # D012
}


def load_bool(path: Path) -> np.ndarray:
    return np.asanyarray(nib.load(str(path)).dataobj) > 0


def dice(gt: np.ndarray, pred: np.ndarray) -> float:
    gt = gt.astype(bool, copy=False)
    pred = pred.astype(bool, copy=False)
    denom = int(gt.sum() + pred.sum())
    if denom == 0:
        return 1.0
    return 2 * int((gt & pred).sum()) / denom


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paths.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["root"])

    df = pd.read_csv(cfg["data"]["test_txt"])
    df["case"] = df["image_path"].map(lambda p: Path(p).name.replace(".nii.gz", ""))

    # "detection" = case where the model found the cyst at all (Dice > 0).
    header = "{:13} {:>3} {:>7} {:>7} {:>6} {:>6} {:>9} {:>6}".format(
        "variant", "n", "mean", "median", "std", "min", "det(>0)", "#<.1")
    print(f"Test set: {len(df)} cases\n")
    print(header)
    print("-" * len(header))

    for name, rel in VARIANT_DIRS.items():
        pdir = root / rel
        scores, missing = [], 0
        for _, r in df.iterrows():
            pf = pdir / f"{r['case']}.nii.gz"
            if not pf.exists():
                missing += 1
                continue
            gt = load_bool(r["mask_path"])
            pred = load_bool(pf)
            if gt.shape != pred.shape:
                print(f"  [shape mismatch] {r['case']}: gt{gt.shape} pred{pred.shape}")
                continue
            scores.append(dice(gt, pred))
        if not scores:
            print(f"{name:13} -- no preds ({missing} missing) at {pdir}")
            continue
        a = np.array(scores)
        det = 100.0 * (a > 0).mean()
        extra = f"  ({missing} missing)" if missing else ""
        print("{:13} {:3d} {:7.4f} {:7.4f} {:6.3f} {:6.3f} {:8.1f}% {:6d}{}".format(
            name, len(a), a.mean(), float(np.median(a)), a.std(), a.min(),
            det, int((a < 0.1).sum()), extra))


if __name__ == "__main__":
    main()
