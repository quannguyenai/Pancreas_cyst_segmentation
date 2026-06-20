"""eval_from_txt.py — Per-case segmentation metrics using the mask_path column
of a split txt (robust to the cyst_<site>_<id> mask naming that the glob-based
evaluate.py misses).

Usage:
  python comparison/eval_from_txt.py \
      --split-txt data/test.txt \
      --pred-dir approach_a/predictions/3d_fullres_5fold \
      --name approach_a_5fold \
      --out-prefix results/approach_a_5fold_test
"""
from __future__ import annotations
import argparse, re
from pathlib import Path
import numpy as np
import nibabel as nib

try:
    from medpy.metric.binary import hd95 as _hd95, asd as _asd
    HAVE_MEDPY = True
except Exception:
    HAVE_MEDPY = False


def site_of(stem: str) -> str:
    m = re.match(r"([A-Za-z]+)", stem)
    return m.group(1).upper() if m else "?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-txt", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--name", default="pred")
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    rows = []
    missing_pred = []
    for line in Path(args.split_txt).read_text().splitlines()[1:]:
        if not line.strip():
            continue
        cols = line.split(",")
        stem = Path(cols[0]).name.replace(".nii.gz", "")
        gt_path = Path(cols[1]) if len(cols) > 1 and cols[1] else None
        pred_path = pred_dir / f"{stem}.nii.gz"
        if not pred_path.exists():
            missing_pred.append(stem)
            continue
        if gt_path is None or not gt_path.exists():
            continue
        gt_nib = nib.load(str(gt_path))
        pred = (nib.load(str(pred_path)).get_fdata() > 0).astype(np.uint8)
        gt = (gt_nib.get_fdata() > 0).astype(np.uint8)
        spacing = tuple(float(z) for z in gt_nib.header.get_zooms()[:3])  # mm/voxel

        tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum()); fn = int((~pred & gt).sum())
        gt_vox = int(gt.sum()); pred_vox = int(pred.sum())
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if gt_vox == 0 else 0.0)
        rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
        hd = asd_v = float("nan")
        if HAVE_MEDPY and pred_vox > 0 and gt_vox > 0:
            try:
                hd = float(_hd95(pred, gt, voxelspacing=spacing))
                asd_v = float(_asd(pred, gt, voxelspacing=spacing))
            except Exception:
                pass
        rows.append(dict(case=stem, site=site_of(stem), gt_vox=gt_vox, pred_vox=pred_vox,
                         dice=dice, precision=prec, recall=rec, iou=iou, hd95=hd, asd=asd_v,
                         empty_gt=(gt_vox == 0)))

    import pandas as pd
    df = pd.DataFrame(rows)
    outp = Path(args.out_prefix)
    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{outp}_per_case.csv", index=False, float_format="%.4f")

    pos = df[~df.empty_gt]  # cases with an actual cyst
    def agg(d):
        return dict(n=len(d), dice_mean=d.dice.mean(), dice_median=d.dice.median(),
                    dice_std=d.dice.std(), recall_mean=d.recall.mean(),
                    precision_mean=d.precision.mean(),
                    hd95_mean=d.hd95.mean(), asd_mean=d.asd.mean(),
                    n_total_miss=int((d.dice < 0.01).sum()))
    print(f"\n=== {args.name} — {len(df)} cases evaluated ({len(pos)} with cyst, "
          f"{len(df)-len(pos)} empty-GT), {len(missing_pred)} preds missing ===")
    s = agg(pos)
    print(f"Dice  mean={s['dice_mean']:.4f}  median={s['dice_median']:.4f}  std={s['dice_std']:.4f}")
    print(f"Recall mean={s['recall_mean']:.4f}   Precision mean={s['precision_mean']:.4f}")
    print(f"HD95  mean={s['hd95_mean']:.2f}   ASD mean={s['asd_mean']:.2f}")
    print(f"Total-miss (Dice<0.01): {s['n_total_miss']} / {len(pos)}")
    print("\nPer-site Dice (cyst cases):")
    site = pos.groupby("site").agg(n=("dice", "size"), dice=("dice", "mean"),
                                   recall=("recall", "mean")).round(4)
    print(site.to_string())
    print("\nWorst 5 (by Dice):")
    print(pos.nsmallest(5, "dice")[["case", "dice", "recall", "gt_vox"]].to_string(index=False))
    print("\nBest 5 (by Dice):")
    print(pos.nlargest(5, "dice")[["case", "dice", "precision", "gt_vox"]].to_string(index=False))

    pd.DataFrame([{"approach": args.name, **agg(pos)}]).to_csv(
        f"{outp}_summary.csv", index=False, float_format="%.4f")
    print(f"\nSaved: {outp}_per_case.csv  and  {outp}_summary.csv")
    if missing_pred:
        print(f"[WARN] missing preds: {missing_pred}")


if __name__ == "__main__":
    main()
