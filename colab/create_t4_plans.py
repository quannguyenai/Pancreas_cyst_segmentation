"""create_t4_plans.py — Generate a Colab T4-compatible nnUNet plans file.

Problem
-------
The auto-planned config uses patch [40, 192, 256] + batch 2 ≈ 14-18 GB VRAM.
Colab T4 has 15 GB VRAM → OOM.

Fix
---
Reduce patch to [40, 128, 192] (2× fewer voxels) → ~7-9 GB VRAM.
Batch size stays at 2. Accuracy loss is minor (~1-2 Dice points).

Usage (run once after nnUNetv2_plan_and_preprocess):
    python colab/create_t4_plans.py --preprocessed-dir /path/to/nnUNet_preprocessed
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


T4_PATCH_SIZE = [40, 128, 192]   # down from [40, 192, 256]
T4_BATCH_SIZE = 2                # keep 2; drop to 1 only if still OOM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create Colab T4-optimised nnUNet plans.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--preprocessed-dir", required=True,
        help="Path to nnUNet_preprocessed/Dataset001_PancreasCyst/",
    )
    p.add_argument("--patch-size", type=int, nargs=3, default=T4_PATCH_SIZE,
                   metavar=("Z", "Y", "X"))
    p.add_argument("--batch-size", type=int, default=T4_BATCH_SIZE)
    p.add_argument("--output-name", default="nnUNetPlans_T4")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    prepd  = Path(args.preprocessed_dir)
    source = prepd / "nnUNetPlans.json"
    output = prepd / f"{args.output_name}.json"

    if not source.exists():
        print(f"[ERROR] Not found: {source}")
        print("        Run nnUNetv2_plan_and_preprocess -d 1 first.")
        sys.exit(1)

    plans   = json.loads(source.read_text())
    t4plans = copy.deepcopy(plans)
    t4plans["plans_name"] = args.output_name

    cfg = t4plans["configurations"]["3d_fullres"]
    orig_patch = cfg["patch_size"][:]
    orig_batch = cfg["batch_size"]

    cfg["patch_size"]      = args.patch_size
    cfg["batch_size"]      = args.batch_size
    cfg["data_identifier"] = "nnUNetPlans_3d_fullres"   # reuse existing preprocessed data

    output.write_text(json.dumps(t4plans, indent=4))

    orig_vol = orig_patch[0] * orig_patch[1] * orig_patch[2]
    new_vol  = args.patch_size[0] * args.patch_size[1] * args.patch_size[2]

    print(f"Written: {output}")
    print(f"  patch  {orig_patch} → {args.patch_size}  ({100*(1-new_vol/orig_vol):.0f}% smaller)")
    print(f"  batch  {orig_batch} → {args.batch_size}")
    print(f"  Est. VRAM: ~{new_vol/orig_vol*17:.0f} GB  (was ~17 GB)")
    print()
    print("To train with this plan:")
    print(f"  nnUNetv2_train 1 3d_fullres 0 -p {args.output_name} --npz")


if __name__ == "__main__":
    main()
