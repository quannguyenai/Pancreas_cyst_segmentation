"""diagnose_normalization.py — measure how much CTNormalization distorts MRI
volumes on Dataset001 compared to ZScoreNormalization (the correct choice
for MRI).

Reads the raw training NIfTIs, applies each normalization scheme, and reports
the per-case foreground statistics *after* normalization. If CT-normalized
cases cluster tightly around (mean=0, std=1), the two schemes are near-
identical in practice and the fold-0 checkpoint is probably fine. If they
spread widely (e.g. std range [0.3, 3.0] across cases), CTNorm left per-case
variance uncorrected and retraining with MRI declaration will help.

Uses *only* the raw NIfTIs + global stats from dataset_fingerprint.json —
does NOT depend on the preprocessed .b2nd files, so it's independent of any
caching issues.

Usage:
    python scripts/diagnose_normalization.py --config configs/paths.yaml
    python scripts/diagnose_normalization.py --config configs/paths.yaml --n 20
"""

from __future__ import annotations

import argparse
import json
import sys
import random
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs import load_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/paths.yaml")
    p.add_argument("--n", type=int, default=40,
                   help="Number of cases to sample (0 = all).")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def ct_normalize(vol: np.ndarray, fg: np.ndarray, stats: dict) -> np.ndarray:
    """nnU-Net v2 CTNormalization: clip to global [p0.5, p99.5], then
    subtract global mean, divide by global std.

    Applied to the whole volume (not masked)."""
    x = np.clip(vol, stats["percentile_00_5"], stats["percentile_99_5"])
    x = (x - stats["mean"]) / stats["std"]
    return x


def zscore_normalize(vol: np.ndarray, fg: np.ndarray) -> np.ndarray:
    """nnU-Net v2 ZScoreNormalization: subtract per-case foreground mean,
    divide by per-case foreground std.

    Foreground here is defined exactly as nnU-Net does it — voxels where
    vol != 0 (a CT-leftover assumption that for MRI still happens to mark
    non-padded regions well because background tends to be exactly 0)."""
    fg_vals = vol[fg]
    if fg_vals.size < 10:
        return vol.astype(np.float32)
    m, s = fg_vals.mean(), fg_vals.std()
    s = max(s, 1e-8)
    return ((vol - m) / s).astype(np.float32)


def foreground_mask(vol: np.ndarray) -> np.ndarray:
    """nnU-Net's default fg heuristic for single-channel input."""
    return vol != 0


def summarize(label: str, values: np.ndarray, key: str) -> str:
    q = np.percentile(values, [0, 25, 50, 75, 100])
    return (f"  {label:>14}  min={q[0]:+.3f}  p25={q[1]:+.3f}  "
            f"med={q[2]:+.3f}  p75={q[3]:+.3f}  max={q[4]:+.3f}  "
            f"(spread {q[4]-q[0]:.2f})")


def main():
    args = parse_args()
    cfg = load_config(args.config)

    fingerprint_path = Path(cfg["nnunet"]["preprocessed"]) / \
        "Dataset001_PancreasCyst" / "dataset_fingerprint.json"
    with open(fingerprint_path) as f:
        fp = json.load(f)
    stats = fp["foreground_intensity_properties_per_channel"]["0"]
    print("=== Global fingerprint stats (used by CTNormalization) ===")
    print(json.dumps(stats, indent=2))
    print()

    # Read the train.txt list (same split used by nnU-Net for Dataset001)
    train_txt = Path(cfg["data"]["train_txt"])
    cases = [r.split(",") for r in train_txt.read_text().splitlines()[1:] if r.strip()]
    random.seed(args.seed)
    if args.n > 0 and args.n < len(cases):
        cases = random.sample(cases, args.n)

    print(f"=== Sampling {len(cases)} training cases ===\n")

    ct_means, ct_stds = [], []
    zs_means, zs_stds = [], []
    raw_means, raw_stds = [], []
    per_case_rows = []

    for img_path_str, _ in cases:
        img_path = Path(img_path_str)
        img = sitk.ReadImage(str(img_path))
        vol = sitk.GetArrayFromImage(img).astype(np.float32)
        fg = foreground_mask(vol)
        if fg.sum() < 1000:
            continue

        raw_fg = vol[fg]
        raw_means.append(raw_fg.mean())
        raw_stds.append(raw_fg.std())

        ct = ct_normalize(vol, fg, stats)
        zs = zscore_normalize(vol, fg)

        ct_fg = ct[fg]
        zs_fg = zs[fg]

        ct_means.append(ct_fg.mean())
        ct_stds.append(ct_fg.std())
        zs_means.append(zs_fg.mean())
        zs_stds.append(zs_fg.std())

        per_case_rows.append((img_path.name.replace(".nii.gz", ""),
                              raw_fg.mean(), raw_fg.std(),
                              ct_fg.mean(), ct_fg.std(),
                              zs_fg.mean(), zs_fg.std()))

    print("=== Per-case foreground intensity distribution ===\n")

    print("Raw intensities (before any normalization):")
    print(summarize("raw means", np.array(raw_means), "mean"))
    print(summarize("raw stds",  np.array(raw_stds),  "std"))
    print()

    print("After CTNormalization (what the checkpoint saw):")
    print(summarize("CT-norm means", np.array(ct_means), "mean"))
    print(summarize("CT-norm stds",  np.array(ct_stds),  "std"))
    print()

    print("After ZScoreNormalization (correct for MRI):")
    print(summarize("z-norm means", np.array(zs_means), "mean"))
    print(summarize("z-norm stds",  np.array(zs_stds),  "std"))
    print()

    # Sample rows for sanity
    print("=== Sample per-case rows (first 10) ===")
    print(f"{'case':<12}  {'raw_μ':>8}  {'raw_σ':>8}  "
          f"{'CT_μ':>7}  {'CT_σ':>7}  {'z_μ':>6}  {'z_σ':>6}")
    for row in per_case_rows[:10]:
        name, rm, rs, cm, cs, zm, zs = row
        print(f"{name:<12}  {rm:>8.1f}  {rs:>8.1f}  "
              f"{cm:>+7.3f}  {cs:>7.3f}  {zm:>+6.3f}  {zs:>6.3f}")
    print()

    # Verdict
    ct_mean_spread = float(np.ptp(ct_means))
    ct_std_range = (float(np.min(ct_stds)), float(np.max(ct_stds)))
    zs_mean_spread = float(np.ptp(zs_means))
    zs_std_range = (float(np.min(zs_stds)), float(np.max(zs_stds)))

    print("=== Verdict ===")
    print(f"CTNorm per-case mean spread: {ct_mean_spread:.3f} "
          f"(ideal: 0.0)")
    print(f"CTNorm per-case std range:   [{ct_std_range[0]:.3f}, {ct_std_range[1]:.3f}] "
          f"(ideal: [1.0, 1.0])")
    print(f"ZScore per-case mean spread: {zs_mean_spread:.3f} (should be ~0)")
    print(f"ZScore per-case std range:   [{zs_std_range[0]:.3f}, {zs_std_range[1]:.3f}] "
          f"(should be ~[1.0, 1.0])")
    print()

    if ct_mean_spread < 0.3 and abs(ct_std_range[1] - ct_std_range[0]) < 0.5:
        verdict = "MINOR — CTNorm happens to behave ~like ZScore on this data. " \
                  "Retraining likely changes Dice by <1%."
    elif ct_mean_spread < 1.0 and abs(ct_std_range[1] - ct_std_range[0]) < 1.5:
        verdict = "MODERATE — noticeable per-case variance left by CTNorm. " \
                  "Retraining may buy 1-3% Dice."
    else:
        verdict = "LARGE — CTNorm leaves massive per-case intensity variation. " \
                  "Retraining is strongly justified; expect >3% Dice improvement."
    print(f"Likely training impact: {verdict}")


if __name__ == "__main__":
    main()
