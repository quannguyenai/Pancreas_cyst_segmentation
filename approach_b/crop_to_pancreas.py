"""crop_to_pancreas.py — Crop CT volumes to pancreas ROI + margin.

For each case, reads the PanSegNet pancreas prediction mask, computes a tight
bounding box, expands it by ``crop_margin_pct`` on each side, then saves
cropped image and cyst mask with a corrected NIfTI affine.

A ``crop_stats.json`` file is written alongside the outputs so that
``paste_back.py`` can invert the transform for full-space evaluation.

Usage
-----
python approach_b/crop_to_pancreas.py --config configs/paths.yaml

# Dry-run (print crop stats, write nothing):
python approach_b/crop_to_pancreas.py --config configs/paths.yaml --dry-run

Notes
-----
* Volumes are reoriented to RAS+ canonical before cropping so that the affine
  update formula is unambiguous. The saved files are in RAS+ space.
* The cyst mask for test cases is not available; only images are cropped for
  test inference. Set --split train/val/test accordingly.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import nibabel.orientations as nio
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs import load_config


# ─── Bounding-box helpers ─────────────────────────────────────────────────────

def compute_tight_bbox(mask: np.ndarray) -> tuple[slice, ...]:
    """Return the smallest bounding box enclosing all nonzero voxels."""
    nonzero = np.argwhere(mask > 0)
    if len(nonzero) == 0:
        return tuple(slice(0, s) for s in mask.shape)
    mins = nonzero.min(axis=0)
    maxs = nonzero.max(axis=0) + 1  # exclusive
    return tuple(slice(int(lo), int(hi)) for lo, hi in zip(mins, maxs))


def expand_bbox(
    bbox: tuple[slice, ...],
    shape: tuple[int, ...],
    margin_pct: float,
) -> tuple[slice, ...]:
    """Expand a bounding box by ``margin_pct`` on each side, clamped to shape."""
    expanded = []
    for sl, dim_size in zip(bbox, shape):
        margin = int(np.ceil(dim_size * margin_pct))
        lo = max(0, sl.start - margin)
        hi = min(dim_size, sl.stop + margin)
        expanded.append(slice(lo, hi))
    return tuple(expanded)


# ─── Crop with affine update ──────────────────────────────────────────────────

def crop_volume(nib_img: nib.Nifti1Image, bbox: tuple[slice, ...]) -> nib.Nifti1Image:
    """Crop a NIfTI image to ``bbox``, updating the affine for the new origin.

    The world-space origin of the cropped volume is:
        new_origin = affine[:3, :3] @ [bbox[i].start for i in range(3)]
                     + affine[:3, 3]
    """
    data    = np.asarray(nib_img.dataobj)
    affine  = nib_img.affine.copy()

    crop_start = np.array([sl.start for sl in bbox], dtype=float)
    new_affine  = affine.copy()
    new_affine[:3, 3] = affine[:3, :3] @ crop_start + affine[:3, 3]

    cropped_data = data[bbox[0], bbox[1], bbox[2]]
    return nib.Nifti1Image(cropped_data, affine=new_affine)


# ─── Per-case pipeline ────────────────────────────────────────────────────────

def process_case(
    stem: str,
    img_path: str,
    mask_path: str | None,
    pancreas_pred_path: str,
    out_images_dir: Path,
    out_masks_dir: Path,
    margin_pct: float,
    dry_run: bool,
) -> dict | None:
    """Crop one case. Returns crop stats dict or None on failure."""
    pancreas_path = Path(pancreas_pred_path)
    if not pancreas_path.exists():
        print(f"[WARN] {stem}: no pancreas prediction at {pancreas_path}, skipping.")
        return None

    img_nib   = nib.load(img_path)
    pan_nib   = nib.load(str(pancreas_path))

    # Reorient both to RAS+ canonical for unambiguous affine math
    img_ras   = nib.as_closest_canonical(img_nib)
    pan_ras   = nib.as_closest_canonical(pan_nib)

    pan_data  = np.asarray(pan_ras.dataobj)

    # Binarise pancreas prediction (may be soft probability map)
    pan_binary = (pan_data > 0.5).astype(np.uint8)

    if pan_binary.sum() == 0:
        print(f"[WARN] {stem}: empty pancreas prediction, skipping.")
        return None

    bbox_tight   = compute_tight_bbox(pan_binary)
    bbox_expanded = expand_bbox(bbox_tight, pan_binary.shape, margin_pct)

    crop_stats = {
        "original_shape": list(np.asarray(img_ras.dataobj).shape),
        "bbox_start": [sl.start for sl in bbox_expanded],
        "bbox_stop":  [sl.stop  for sl in bbox_expanded],
    }

    if not dry_run:
        out_images_dir.mkdir(parents=True, exist_ok=True)
        cropped_img = crop_volume(img_ras, bbox_expanded)
        nib.save(cropped_img, str(out_images_dir / f"{stem}.nii.gz"))

        if mask_path and Path(mask_path).exists():
            out_masks_dir.mkdir(parents=True, exist_ok=True)
            mask_nib = nib.load(mask_path)
            mask_ras = nib.as_closest_canonical(mask_nib)
            cropped_mask = crop_volume(mask_ras, bbox_expanded)
            nib.save(cropped_mask, str(out_masks_dir / f"{stem}.nii.gz"))

    orig_vol = int(np.prod(crop_stats["original_shape"]))
    crop_vol = int(np.prod([
        s - t for s, t in zip(crop_stats["bbox_stop"], crop_stats["bbox_start"])
    ]))
    reduction = 100 * (1 - crop_vol / orig_vol)
    print(f"  {stem}: {crop_stats['original_shape']} → "
          f"{[s-t for s,t in zip(crop_stats['bbox_stop'], crop_stats['bbox_start'])]} "
          f"({reduction:.1f}% reduction)")

    return {stem: crop_stats}


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Crop CT volumes to pancreas ROI + margin.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="configs/paths.yaml")
    p.add_argument("--split", default="all",
                   choices=["train", "val", "test", "all"],
                   help="Which split to crop. 'all' processes train+val+test.")
    p.add_argument("--workers", type=int, default=4,
                   help="Number of parallel worker processes.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print crop statistics without writing any files.")
    return p.parse_args()


def load_split(cfg: dict, split: str) -> list[tuple[str, str | None]]:
    """Return [(img_path, mask_path_or_None), ...] for a split."""
    key_map = {"train": "train_txt", "val": "val_txt", "test": "test_txt"}
    txt = Path(cfg["data"][key_map[split]])
    rows = []
    with open(txt) as f:
        for line in f.readlines()[1:]:
            line = line.strip()
            if line:
                parts = line.split(",")
                img   = parts[0]
                mask  = parts[1] if len(parts) > 1 else None
                rows.append((img, mask))
    return rows


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)

    margin_pct    = float(cfg["approach_b"]["crop_margin_pct"])
    pan_preds_dir = Path(cfg["approach_b"]["pancreas_preds"])
    out_images    = Path(cfg["approach_b"]["cropped_images"])
    out_masks     = Path(cfg["approach_b"]["cropped_masks"])
    stats_path    = Path(cfg["approach_b"]["crop_stats_json"])

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    all_cases: list[tuple[str, str | None]] = []
    for sp in splits:
        all_cases.extend(load_split(cfg, sp))

    print(f"Cropping {len(all_cases)} cases (margin={margin_pct*100:.0f}%)"
          + (" [DRY RUN]" if args.dry_run else ""))

    crop_stats_all: dict[str, dict] = {}

    futures_map = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for img_path, mask_path in all_cases:
            stem = Path(img_path).name.replace(".nii.gz", "")
            # PanSegNet outputs may keep the original stem or add _0000
            pan_pred = pan_preds_dir / f"{stem}.nii.gz"
            if not pan_pred.exists():
                pan_pred = pan_preds_dir / f"{stem}_0000.nii.gz"

            fut = pool.submit(
                process_case,
                stem, img_path, mask_path,
                str(pan_pred),
                out_images, out_masks,
                margin_pct, args.dry_run,
            )
            futures_map[fut] = stem

        for fut in as_completed(futures_map):
            result = fut.result()
            if result:
                crop_stats_all.update(result)

    if not args.dry_run:
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(crop_stats_all, indent=2) + "\n")
        print(f"\nCrop stats written to: {stats_path}")
        print(f"Successfully cropped:  {len(crop_stats_all)}/{len(all_cases)} cases")


if __name__ == "__main__":
    main()
