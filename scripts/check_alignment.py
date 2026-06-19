"""check_alignment.py — verify the user's categorisation of mask/image alignment
into two groups:

  Group A: CAD cases — mask direction is identity (1,0,0;0,1,0;0,0,1) while
           image direction is real and non-trivial. O(1) discrepancy. Real
           training impact. Need to overwrite mask affine with image affine.

  Group B: EMC068 / IU52 / NU108 / NU118 — direction matrices match modulo
           ~1e-5 float noise. No training impact; only strict-equality check
           failures.

Reports for each case:
  - image direction matrix (3 rows)
  - mask direction matrix (3 rows)
  - max element-wise abs difference
  - max origin diff (mm)
  - classification
"""

from __future__ import annotations

import sys
import argparse
import re
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).parent.parent))


_INST_PATTERN = re.compile(r"^([A-Za-z]+?)(\d+)$")


def image_stem_to_mask_stem(stem: str) -> str:
    m = _INST_PATTERN.match(stem)
    if not m:
        raise ValueError(f"Cannot parse image stem: {stem!r}")
    return f"cyst_{m.group(1).lower()}_{m.group(2)}"


def load_metadata(path: Path) -> dict:
    img = sitk.ReadImage(str(path))
    return {
        "direction": np.array(img.GetDirection()).reshape(3, 3),
        "origin":    np.array(img.GetOrigin()),
        "spacing":   np.array(img.GetSpacing()),
        "size":      img.GetSize(),
    }


def classify(img_meta: dict, mask_meta: dict) -> tuple[str, dict]:
    img_dir, mask_dir = img_meta["direction"], mask_meta["direction"]
    diff = np.abs(img_dir - mask_dir)
    max_dir_diff = float(diff.max())

    origin_diff = float(np.abs(img_meta["origin"] - mask_meta["origin"]).max())
    spacing_diff = float(np.abs(img_meta["spacing"] - mask_meta["spacing"]).max())

    is_mask_identity = np.allclose(mask_dir, np.eye(3), atol=1e-6)
    is_img_identity  = np.allclose(img_dir,  np.eye(3), atol=1e-6)

    if is_mask_identity and not is_img_identity:
        klass = "BROKEN — mask is identity, image is real"
    elif max_dir_diff < 1e-3 and origin_diff < 1e-2:
        klass = "OK — sub-voxel float noise only"
    elif max_dir_diff < 1e-2 and origin_diff < 1e-1:
        klass = "MARGINAL — small drift"
    else:
        klass = "BROKEN — large direction/origin mismatch"

    return klass, {
        "max_dir_diff": max_dir_diff,
        "origin_diff_mm": origin_diff,
        "spacing_diff": spacing_diff,
        "mask_is_identity": is_mask_identity,
        "img_is_identity": is_img_identity,
    }


def fmt_matrix(m: np.ndarray) -> str:
    return "\n  ".join("[" + "  ".join(f"{v:+.6f}" for v in row) + "]"
                       for row in m)


def inspect(stem: str, images_dir: Path, masks_dir: Path,
            verbose: bool = False) -> tuple[str, dict]:
    img_path = images_dir / f"{stem}.nii.gz"
    mask_path = masks_dir / f"{image_stem_to_mask_stem(stem)}.nii.gz"
    if not img_path.exists():
        return "MISSING", {"reason": f"no image at {img_path}"}
    if not mask_path.exists():
        return "MISSING", {"reason": f"no mask at {mask_path}"}

    img_meta = load_metadata(img_path)
    mask_meta = load_metadata(mask_path)
    klass, stats = classify(img_meta, mask_meta)

    if verbose:
        print(f"\n=== {stem} ===")
        print(f"  classification: {klass}")
        print(f"  max direction diff: {stats['max_dir_diff']:.3e}")
        print(f"  origin diff (mm):   {stats['origin_diff_mm']:.3e}")
        print(f"  spacing diff:       {stats['spacing_diff']:.3e}")
        print(f"  mask is identity:   {stats['mask_is_identity']}")
        print(f"  image is identity:  {stats['img_is_identity']}")
        print(f"  IMAGE direction:")
        print(f"  {fmt_matrix(img_meta['direction'])}")
        print(f"  MASK direction:")
        print(f"  {fmt_matrix(mask_meta['direction'])}")
    return klass, stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data")
    p.add_argument("--cases", nargs="*", default=None,
                   help="Specific case stems to inspect verbosely. "
                        "If omitted: scans all CAD + EMC068/IU52/NU108/NU118.")
    args = p.parse_args()

    images_dir = Path(args.data_root) / "images"
    masks_dir  = Path(args.data_root) / "masks"

    if args.cases:
        for stem in args.cases:
            inspect(stem, images_dir, masks_dir, verbose=True)
        return

    # Scan: all CAD + the specific EMC/IU/NU cases.
    cad_stems = sorted(p.name.replace(".nii.gz", "") for p in images_dir.glob("CAD*.nii.gz"))
    target_stems = ["EMC068", "IU52", "NU108", "NU118"]

    print(f"=== Group A: {len(cad_stems)} CAD cases ===")
    cad_classes = {}
    for stem in cad_stems:
        klass, stats = inspect(stem, images_dir, masks_dir, verbose=False)
        cad_classes.setdefault(klass, []).append((stem, stats))
    for klass, items in cad_classes.items():
        print(f"\n  [{klass}]  ({len(items)} cases)")
        for stem, stats in items[:5]:
            print(f"    {stem:<10}  max_dir_diff={stats.get('max_dir_diff', 'n/a'):>11}  "
                  f"origin_diff={stats.get('origin_diff_mm', 'n/a'):>11}")
        if len(items) > 5:
            print(f"    ... and {len(items) - 5} more")

    print(f"\n=== Group B: {target_stems} ===")
    for stem in target_stems:
        klass, stats = inspect(stem, images_dir, masks_dir, verbose=True)


if __name__ == "__main__":
    main()
