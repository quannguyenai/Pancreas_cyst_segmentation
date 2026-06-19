"""shift_for_predict.py — Expand single-channel MRI volumes into the 5-channel
stack-as-channels 2.5D form expected by the Dataset011_PancreasCyst25D model
at inference time.

Reads each `<stem>.nii.gz` (or `<stem>_0000.nii.gz`) from --input-dir and writes
`<stem>_0000.nii.gz .. _{window-1:04d}.nii.gz` to --output-dir, where channel k
is the volume z-shifted by (k - window//2) with replicate padding at boundaries.

Usage
-----
python approach_a/shift_for_predict.py \\
    --input-dir  /path/to/raw_mri \\
    --output-dir /path/to/stacked

This is what predict_stack5_raw.sh calls. You can also invoke it directly
before running nnUNetv2_predict with -d 11.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).parent))
from prepare_stack5_dataset import write_shifted_channels  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Expand raw single-channel MRI volumes into the 5-channel "
                    "stack-as-channels form for Dataset011 inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-dir", required=True, type=Path,
                   help="Folder of single-channel MRI NIfTIs (.nii.gz). "
                        "Files may be named <stem>.nii.gz or <stem>_0000.nii.gz.")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Folder to write {stem}_0000.nii.gz .. _000{W-1}.nii.gz.")
    p.add_argument("--window", type=int, default=5, choices=[3, 5])
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def strip_channel_suffix(name: str) -> str:
    """<stem>_0000.nii.gz → <stem>. Leaves <stem>.nii.gz → <stem>."""
    base = name.replace(".nii.gz", "")
    if len(base) > 5 and base[-5] == "_" and base[-4:].isdigit():
        return base[:-5]
    return base


def main() -> None:
    args = parse_args()
    if args.window % 2 == 0:
        raise SystemExit(f"--window must be odd, got {args.window}")

    half = args.window // 2
    shifts = list(range(-half, half + 1))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    inputs = sorted(args.input_dir.glob("*.nii.gz"))
    if not inputs:
        raise SystemExit(f"No .nii.gz files found under {args.input_dir}")

    # Deduplicate by stem — if both <stem>.nii.gz and <stem>_0000.nii.gz exist,
    # prefer the _0000 form (matches nnU-Net's imagesTs layout).
    by_stem: dict[str, Path] = {}
    for p in inputs:
        stem = strip_channel_suffix(p.name)
        if stem not in by_stem or p.name.endswith("_0000.nii.gz"):
            by_stem[stem] = p

    print(f"[INFO] Shifting {len(by_stem)} cases  window={args.window} shifts={shifts}")
    print(f"       input : {args.input_dir}")
    print(f"       output: {args.output_dir}")

    for i, (stem, img_path) in enumerate(sorted(by_stem.items()), 1):
        write_shifted_channels(img_path, args.output_dir, stem, shifts, args.force)
        if i % 10 == 0 or i == len(by_stem):
            print(f"       {i}/{len(by_stem)}")


if __name__ == "__main__":
    main()
