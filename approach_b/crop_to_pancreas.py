"""crop_to_pancreas.py — Crop CT volumes to pancreas ROI + margin.

For each case, reads the pancreas prediction mask, computes a tight
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
* Cropping is performed in the original voxel grid of each image, not after a
  canonical reorientation. This preserves direct paste-back into full space.
* If the pancreas prediction or cyst mask header does not match the image, it
  is resampled onto the image grid before cropping.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import nibabel.processing as nibp
import numpy as np
from nibabel.affines import voxel_sizes

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
        extent = max(1, sl.stop - sl.start)
        margin = int(np.ceil(extent * margin_pct))
        lo = max(0, sl.start - margin)
        hi = min(dim_size, sl.stop + margin)
        expanded.append(slice(lo, hi))
    return tuple(expanded)


def compute_centroid_box(
    mask: np.ndarray,
    shape: tuple[int, ...],
    box_mm: tuple[float, ...],
    spacing: tuple[float, ...],
) -> tuple[slice, ...]:
    """Fixed-size box (``box_mm`` per axis) centered on the pancreas centroid.

    The physical extent is converted to voxels per axis via ``spacing``. When the
    centered box would run off a volume edge it is *shifted inward* to preserve
    the full extent (rather than padded), so every case with a large-enough volume
    gets an identically sized crop. Only when a volume axis is smaller than the
    requested extent does that axis shrink to the volume size. This keeps the
    crop shape equal to ``bbox_stop - bbox_start`` so ``paste_back.py`` needs no
    change.
    """
    nonzero = np.argwhere(mask > 0)
    if len(nonzero) == 0:
        return tuple(slice(0, s) for s in shape)
    centroid = nonzero.mean(axis=0)
    box = []
    for c, dim_size, mm, sp in zip(centroid, shape, box_mm, spacing):
        extent = max(1, min(int(round(mm / sp)), int(dim_size)))
        start = int(round(c - extent / 2.0))
        start = max(0, min(start, int(dim_size) - extent))  # shift inward
        box.append(slice(start, start + extent))
    return tuple(box)


# ─── Pancreas-prior (distance map) channel ────────────────────────────────────

def pancreas_distance_map(
    pan_crop: np.ndarray,
    spacing: tuple[float, ...],
    clip_mm: float = 40.0,
) -> np.ndarray:
    """Signed, clamped, normalized distance map of the pancreas mask.

    Positive inside the pancreas, negative outside, in millimetres clamped to
    ``±clip_mm`` and scaled to [-1, 1]. A smooth distance field is a stronger
    localization prior for the network than a hard 0/1 mask.
    """
    from scipy.ndimage import distance_transform_edt

    fg = pan_crop > 0
    if not fg.any():
        return np.zeros(pan_crop.shape, dtype=np.float32)
    dist_out = distance_transform_edt(~fg, sampling=spacing)
    dist_in = distance_transform_edt(fg, sampling=spacing)
    signed = dist_in - dist_out  # mm: + inside, - outside
    signed = np.clip(signed, -clip_mm, clip_mm) / clip_mm
    return signed.astype(np.float32)


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
    mode: str = "tight-margin",
    box_mm: tuple[float, ...] | None = None,
    emit_distance: bool = False,
    out_prior_dir: Path | None = None,
) -> dict | None:
    """Crop one case. Returns crop stats dict (with cyst-fit info) or None on failure."""
    pancreas_path = Path(pancreas_pred_path)
    if not pancreas_path.exists():
        print(f"[WARN] {stem}: no pancreas prediction at {pancreas_path}, skipping.")
        return None

    img_nib = nib.load(img_path)
    pan_nib = nib.load(str(pancreas_path))

    if pan_nib.shape != img_nib.shape or not np.allclose(pan_nib.affine, img_nib.affine, atol=1e-3):
        pan_nib = nibp.resample_from_to(pan_nib, img_nib, order=0)

    pan_data = np.asarray(pan_nib.dataobj)
    pan_binary = (pan_data > 0).astype(np.uint8)

    if pan_binary.sum() == 0:
        print(f"[WARN] {stem}: empty pancreas prediction, skipping.")
        return None

    spacing = tuple(voxel_sizes(img_nib.affine))

    if mode == "fixed-box":
        bbox = compute_centroid_box(pan_binary, pan_binary.shape, box_mm, spacing)
    else:
        bbox_tight = compute_tight_bbox(pan_binary)
        bbox = expand_bbox(bbox_tight, pan_binary.shape, margin_pct)

    crop_stats = {
        "original_shape": list(img_nib.shape),
        "bbox_start": [sl.start for sl in bbox],
        "bbox_stop":  [sl.stop  for sl in bbox],
    }

    # Load the cyst GT once (needed for both cropping and the dry-run fit check).
    mask_nib = None
    if mask_path and Path(mask_path).exists():
        mask_nib = nib.load(mask_path)
        if mask_nib.shape != img_nib.shape or not np.allclose(mask_nib.affine, img_nib.affine, atol=1e-3):
            mask_nib = nibp.resample_from_to(mask_nib, img_nib, order=0)

    # Does the cyst GT fit entirely inside the crop box? (clipping diagnostic)
    # None = no cyst voxels in this case; True/False = fits / clipped.
    cyst_fits: bool | None = None
    if mask_nib is not None:
        cyst_binary = (np.asarray(mask_nib.dataobj) > 0).astype(np.uint8)
        if cyst_binary.sum() > 0:
            cyst_bbox = compute_tight_bbox(cyst_binary)
            cyst_fits = all(
                cb.start >= sl.start and cb.stop <= sl.stop
                for cb, sl in zip(cyst_bbox, bbox)
            )
    crop_stats["cyst_fits"] = cyst_fits

    if not dry_run:
        out_images_dir.mkdir(parents=True, exist_ok=True)
        cropped_img = crop_volume(img_nib, bbox)
        nib.save(cropped_img, str(out_images_dir / f"{stem}.nii.gz"))

        if emit_distance and out_prior_dir is not None:
            out_prior_dir.mkdir(parents=True, exist_ok=True)
            pan_crop = pan_binary[bbox[0], bbox[1], bbox[2]]
            dist = pancreas_distance_map(pan_crop, spacing)
            nib.save(
                nib.Nifti1Image(dist, affine=cropped_img.affine),
                str(out_prior_dir / f"{stem}.nii.gz"),
            )

        if mask_nib is not None:
            out_masks_dir.mkdir(parents=True, exist_ok=True)
            cropped_mask = crop_volume(mask_nib, bbox)
            cropped_mask = nib.Nifti1Image(
                np.asarray(cropped_mask.dataobj).astype(np.uint8),
                affine=cropped_mask.affine,
            )
            nib.save(cropped_mask, str(out_masks_dir / f"{stem}.nii.gz"))

    orig_vol = int(np.prod(crop_stats["original_shape"]))
    crop_vol = int(np.prod([
        s - t for s, t in zip(crop_stats["bbox_stop"], crop_stats["bbox_start"])
    ]))
    reduction = 100 * (1 - crop_vol / orig_vol)
    fit_str = "" if cyst_fits is None else (" cyst:OK" if cyst_fits else " cyst:CLIPPED")
    print(f"  {stem}: {crop_stats['original_shape']} → "
          f"{[s-t for s,t in zip(crop_stats['bbox_stop'], crop_stats['bbox_start'])]} "
          f"({reduction:.1f}% reduction){fit_str}")

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
    p.add_argument(
        "--stats-path", default=None,
        help="Override output path for crop_stats.json.",
    )
    p.add_argument(
        "--mode", default=None, choices=["tight-margin", "fixed-box"],
        help="Crop mode. Defaults to approach_b.crop_mode in the config.",
    )
    p.add_argument(
        "--emit-distance-channel", action="store_true",
        help="Also write a pancreas distance-map crop (for the 2-channel variant).",
    )
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

    ab            = cfg["approach_b"]
    margin_pct    = float(ab["crop_margin_pct"])
    pan_preds_dir = Path(ab["pancreas_preds"])
    out_images    = Path(ab["cropped_images"])
    out_masks     = Path(ab["cropped_masks"])
    stats_path    = Path(args.stats_path or ab["crop_stats_json"])

    mode    = args.mode or ab.get("crop_mode", "tight-margin")
    box_mm  = tuple(float(x) for x in ab.get("crop_box_mm", [64, 192, 192]))
    out_prior = Path(ab["cropped_prior"]) if ab.get("cropped_prior") else None
    emit_distance = args.emit_distance_channel

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    all_cases: list[tuple[str, str | None]] = []
    for sp in splits:
        all_cases.extend(load_split(cfg, sp))

    detail = (f"mode={mode}, box_mm={list(box_mm)}" if mode == "fixed-box"
              else f"mode={mode}, margin={margin_pct*100:.0f}%")
    print(f"Cropping {len(all_cases)} cases ({detail})"
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
                mode, box_mm, emit_distance, out_prior,
            )
            futures_map[fut] = stem

        for fut in as_completed(futures_map):
            result = fut.result()
            if result:
                crop_stats_all.update(result)

    # Cyst-clipping summary (diagnostic for tuning crop_box_mm).
    with_cyst = [s for s in crop_stats_all.values() if s.get("cyst_fits") is not None]
    if with_cyst:
        clipped = [s for s in with_cyst if not s["cyst_fits"]]
        print(f"\nCyst fit: {len(with_cyst) - len(clipped)}/{len(with_cyst)} contained, "
              f"{len(clipped)} CLIPPED.")

    if not args.dry_run:
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(crop_stats_all, indent=2) + "\n")
        print(f"\nCrop stats written to: {stats_path}")
        print(f"Successfully cropped:  {len(crop_stats_all)}/{len(all_cases)} cases")


if __name__ == "__main__":
    main()
