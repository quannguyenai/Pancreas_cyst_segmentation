"""paste_back.py — Invert the crop transform to restore full-space predictions.

After nnUNet inference on cropped volumes, this script zero-pads each
predicted mask back into the original image space using the bounding box
recorded in ``crop_stats.json``.

Usage
-----
python approach_b/paste_back.py --config configs/paths.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Paste cropped predictions back into original image space.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="configs/paths.yaml")
    p.add_argument(
        "--cropped-preds", default=None,
        help="Override path to cropped prediction NIfTIs (default from config).",
    )
    p.add_argument(
        "--output", default=None,
        help="Override output directory for full-space predictions.",
    )
    return p.parse_args()


def paste_back(
    pred_data: np.ndarray,
    bbox_start: list[int],
    bbox_stop: list[int],
    original_shape: list[int],
) -> np.ndarray:
    """Zero-pad ``pred_data`` into an array of ``original_shape``."""
    full = np.zeros(original_shape, dtype=pred_data.dtype)
    slices = tuple(
        slice(start, stop)
        for start, stop in zip(bbox_start, bbox_stop)
    )
    expected = tuple(stop - start for start, stop in zip(bbox_start, bbox_stop))
    if pred_data.shape != expected:
        # Resize to match expected crop shape (handle minor nnUNet rounding)
        from skimage.transform import resize
        pred_data = resize(
            pred_data.astype(float), expected,
            order=0, preserve_range=True, anti_aliasing=False,
        ).astype(pred_data.dtype)
    full[slices] = pred_data
    return full


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)

    stats_path    = Path(cfg["approach_b"]["crop_stats_json"])
    cropped_preds = Path(args.cropped_preds or cfg["approach_b"]["predictions_cropped"])
    output_dir    = Path(args.output or cfg["approach_b"]["predictions_full"])
    images_dir    = Path(cfg["data"]["images"])

    if not stats_path.exists():
        print(f"[ERROR] crop_stats.json not found at {stats_path}")
        print("        Run crop_to_pancreas.py first.")
        sys.exit(1)

    crop_stats: dict = json.loads(stats_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)

    ok, failed = 0, 0
    for stem, stats in crop_stats.items():
        pred_path = cropped_preds / f"{stem}.nii.gz"
        if not pred_path.exists():
            print(f"[SKIP] {stem}: no cropped prediction found.")
            failed += 1
            continue

        pred_nib = nib.load(str(pred_path))
        pred_data = np.asarray(pred_nib.dataobj).astype(np.uint8)

        full_pred = paste_back(
            pred_data,
            stats["bbox_start"],
            stats["bbox_stop"],
            stats["original_shape"],
        )

        # Use the original image affine for the output
        img_path = images_dir / f"{stem}.nii.gz"
        if img_path.exists():
            ref_affine = nib.load(str(img_path)).affine
        else:
            # Reconstruct affine from crop: reverse the crop shift
            ref_affine = pred_nib.affine.copy()
            crop_start = np.array(stats["bbox_start"], dtype=float)
            ref_affine[:3, 3] -= ref_affine[:3, :3] @ crop_start

        out_nib = nib.Nifti1Image(full_pred, affine=ref_affine)
        nib.save(out_nib, str(output_dir / f"{stem}.nii.gz"))
        ok += 1

    print(f"Paste-back complete: {ok} succeeded, {failed} skipped.")
    print(f"Full-space predictions: {output_dir}")


if __name__ == "__main__":
    main()
