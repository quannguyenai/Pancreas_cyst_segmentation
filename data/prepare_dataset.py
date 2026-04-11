"""prepare_dataset.py — Convert raw NIfTI data to nnUNet format and update split CSVs.

Usage
-----
# Fix CAD header mismatches and update split CSVs to use repo-relative paths:
python data/prepare_dataset.py --config configs/paths.yaml --fix-cad-headers --update-txts

# Rebuild nnUNet Dataset001 symlinks without touching split CSVs:
python data/prepare_dataset.py --config configs/paths.yaml --build-nnunet

# Full setup (recommended for fresh clone):
python data/prepare_dataset.py --config configs/paths.yaml \\
    --fix-cad-headers --update-txts --build-nnunet

Notes
-----
- The 247/37/74 split is NOT re-randomised; existing case stems are preserved.
- CAD masks have identity direction matrices (known issue logged during nnUNet
  preprocessing); --fix-cad-headers copies the affine from the image onto the mask.
- Requires: nibabel, numpy, pyyaml (all in requirements.txt).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

# Allow running as a script without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs import load_config


# ─── Naming convention helpers ────────────────────────────────────────────────

# Image stems look like:  EMC024, IU9, NU51, AHN05, CAD102, MCF3, NYU12, MCA5
# Mask  stems look like:  cyst_emc_024, cyst_iu_9, cyst_nu_51, ...
# The pattern: lowercase institution prefix + zero-padded or bare number.
_INST_PATTERN = re.compile(r"^([A-Za-z]+?)(\d+)$")


def image_stem_to_mask_stem(stem: str) -> str:
    """Convert an image stem to its expected mask stem.

    Examples
    --------
    >>> image_stem_to_mask_stem("EMC024")
    'cyst_emc_024'
    >>> image_stem_to_mask_stem("IU9")
    'cyst_iu_9'
    >>> image_stem_to_mask_stem("CAD102")
    'cyst_cad_102'
    """
    m = _INST_PATTERN.match(stem)
    if not m:
        raise ValueError(f"Cannot parse image stem: {stem!r}")
    prefix, number = m.group(1).lower(), m.group(2)
    return f"cyst_{prefix}_{number}"


def discover_cases(images_dir: Path, masks_dir: Path) -> dict[str, tuple[Path, Path]]:
    """Scan images_dir for *.nii.gz files and match each to its mask.

    Returns
    -------
    dict mapping case_id (image stem) → (image_path, mask_path)
    """
    cases: dict[str, tuple[Path, Path]] = {}
    missing: list[str] = []

    for img_path in sorted(images_dir.glob("*.nii.gz")):
        stem = img_path.name.replace(".nii.gz", "")
        mask_stem = image_stem_to_mask_stem(stem)
        mask_path = masks_dir / f"{mask_stem}.nii.gz"
        if mask_path.exists():
            cases[stem] = (img_path, mask_path)
        else:
            missing.append(f"  {stem} → expected {mask_path}")

    if missing:
        print(f"[WARN] {len(missing)} images have no matching mask:")
        for m in missing[:10]:
            print(m)
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    print(f"[INFO] Discovered {len(cases)} complete image-mask pairs.")
    return cases


# ─── Split CSV helpers ────────────────────────────────────────────────────────

def update_split_txts(
    data_dir: Path,
    images_dir: Path,
    masks_dir: Path,
    cases: dict[str, tuple[Path, Path]],
) -> None:
    """Rewrite train/val/test/all_train .txt files with updated paths.

    Reads the existing split files to preserve the original case assignment,
    then replaces each absolute path with the new images_dir/masks_dir location.
    Does NOT re-randomise the split.
    """
    split_files = {
        "train.txt": data_dir / "train.txt",
        "val.txt":   data_dir / "val.txt",
        "test.txt":  data_dir / "test.txt",
        "all_train.txt": data_dir / "all_train.txt",
    }

    # Build a lookup: image stem → (new_img_path, new_mask_path)
    new_paths: dict[str, tuple[Path, Path]] = {}
    for stem, (img_p, mask_p) in cases.items():
        new_paths[stem] = (img_p, mask_p)

    for fname, txt_path in split_files.items():
        if not txt_path.exists():
            print(f"[SKIP] {fname} not found, skipping.")
            continue

        lines = txt_path.read_text().splitlines()
        header = lines[0]  # "image_path,mask_path"
        rows = [line.strip() for line in lines[1:] if line.strip()]

        updated: list[str] = [header]
        not_found: list[str] = []

        for row in rows:
            # Extract the case stem from the image path (last path component)
            old_img_path = row.split(",")[0]
            old_stem = Path(old_img_path).name.replace(".nii.gz", "")
            if old_stem in new_paths:
                new_img, new_mask = new_paths[old_stem]
                updated.append(f"{new_img},{new_mask}")
            else:
                not_found.append(old_stem)
                updated.append(row)  # keep old row unchanged

        txt_path.write_text("\n".join(updated) + "\n")
        n_updated = len(updated) - 1 - len(not_found)
        print(f"[INFO] {fname}: updated {n_updated}/{len(rows)} rows"
              + (f", {len(not_found)} not found in images_dir" if not_found else ""))


# ─── nnUNet dataset construction ──────────────────────────────────────────────

DATASET_JSON_TEMPLATE = {
    "channel_names": {"0": "CT"},
    "labels": {"background": 0, "cyst": 1},
    "numTraining": 0,
    "file_ending": ".nii.gz",
    "overwrite_image_reader_writer": "NibabelIOWithReorient",
}


def build_nnunet_raw(
    cases: dict[str, tuple[Path, Path]],
    train_txt: Path,
    test_txt: Path,
    nnunet_raw_dir: Path,
    dataset_id: int,
    dataset_name: str,
) -> None:
    """Create nnUNet Dataset folder structure with symlinks.

    Structure created:
        nnUNet_raw/Dataset{id:03d}_{name}/
            imagesTr/CASE_0000.nii.gz  (symlink)
            labelsTr/CASE.nii.gz        (symlink)
            imagesTs/CASE_0000.nii.gz   (symlink, test images only)
            dataset.json
    """
    dataset_folder = nnunet_raw_dir / f"Dataset{dataset_id:03d}_{dataset_name}"
    images_tr = dataset_folder / "imagesTr"
    labels_tr  = dataset_folder / "labelsTr"
    images_ts  = dataset_folder / "imagesTs"

    for d in (images_tr, labels_tr, images_ts):
        d.mkdir(parents=True, exist_ok=True)

    # Read training case stems from train.txt
    train_stems: set[str] = set()
    if train_txt.exists():
        for line in train_txt.read_text().splitlines()[1:]:
            line = line.strip()
            if line:
                stem = Path(line.split(",")[0]).name.replace(".nii.gz", "")
                train_stems.add(stem)

    test_stems: set[str] = set()
    if test_txt.exists():
        for line in test_txt.read_text().splitlines()[1:]:
            line = line.strip()
            if line:
                stem = Path(line.split(",")[0]).name.replace(".nii.gz", "")
                test_stems.add(stem)

    n_train = 0
    for stem, (img_path, mask_path) in cases.items():
        if stem in train_stems:
            dst_img  = images_tr / f"{stem}_0000.nii.gz"
            dst_mask = labels_tr  / f"{stem}.nii.gz"
            for dst, src in [(dst_img, img_path), (dst_mask, mask_path)]:
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                dst.symlink_to(src.resolve())
            n_train += 1
        elif stem in test_stems:
            dst_img = images_ts / f"{stem}_0000.nii.gz"
            if dst_img.exists() or dst_img.is_symlink():
                dst_img.unlink()
            dst_img.symlink_to(img_path.resolve())

    # Write dataset.json
    dataset_json = {**DATASET_JSON_TEMPLATE}
    dataset_json["numTraining"] = n_train
    dataset_json["name"] = dataset_name
    (dataset_folder / "dataset.json").write_text(
        json.dumps(dataset_json, indent=4) + "\n"
    )

    print(f"[INFO] Built {dataset_folder.name}: {n_train} training cases, "
          f"{len(test_stems)} test cases.")


# ─── CAD header fix ───────────────────────────────────────────────────────────

def fix_cad_headers(images_dir: Path, masks_dir: Path) -> None:
    """Copy image affine onto CAD masks that have identity direction matrices.

    During nnUNet preprocessing, warnings were raised for CAD cases due to
    mismatched origins/directions between images and masks. This function
    overwrites the mask affine with the image affine (in-place, with backup).
    """
    cad_images = sorted(images_dir.glob("CAD*.nii.gz"))
    fixed = 0
    for img_path in cad_images:
        stem = img_path.name.replace(".nii.gz", "")
        mask_stem = image_stem_to_mask_stem(stem)
        mask_path = masks_dir / f"{mask_stem}.nii.gz"
        if not mask_path.exists():
            continue

        img_nib  = nib.load(str(img_path))
        mask_nib = nib.load(str(mask_path))

        img_affine  = img_nib.affine
        mask_affine = mask_nib.affine

        if np.allclose(img_affine, mask_affine, atol=1e-3):
            continue  # already aligned

        # Backup original mask
        backup = mask_path.with_suffix(".bak.nii.gz")
        if not backup.exists():
            shutil.copy2(mask_path, backup)

        fixed_nib = nib.Nifti1Image(
            mask_nib.get_fdata().astype(np.uint8),
            affine=img_affine,
            header=img_nib.header,
        )
        nib.save(fixed_nib, str(mask_path))
        fixed += 1

    print(f"[INFO] Fixed affine for {fixed}/{len(cad_images)} CAD masks "
          "(originals backed up as .bak.nii.gz).")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare the Pancreas Cyst dataset for training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", default="configs/paths.yaml",
        help="Path to configs/paths.yaml",
    )
    p.add_argument(
        "--fix-cad-headers", action="store_true",
        help="Copy image affine onto CAD masks with mismatched headers.",
    )
    p.add_argument(
        "--update-txts", action="store_true",
        help="Rewrite train/val/test/all_train .txt files with current paths.",
    )
    p.add_argument(
        "--build-nnunet", action="store_true",
        help="Create nnUNet_raw/Dataset001_PancreasCyst symlink tree.",
    )
    p.add_argument(
        "--dataset-id", type=int, default=None,
        help="Override nnunet.dataset_id from config (for building extra datasets).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    images_dir = Path(cfg["data"]["images"])
    masks_dir  = Path(cfg["data"]["masks"])
    data_dir   = Path(cfg["root"]) / "data"

    if not images_dir.exists():
        print(f"[ERROR] images_dir not found: {images_dir}")
        print("        Follow the data access instructions in README.md first.")
        sys.exit(1)

    cases = discover_cases(images_dir, masks_dir)

    if args.fix_cad_headers:
        fix_cad_headers(images_dir, masks_dir)

    if args.update_txts:
        update_split_txts(data_dir, images_dir, masks_dir, cases)

    if args.build_nnunet:
        dataset_id = args.dataset_id or int(cfg["nnunet"]["dataset_id"])
        dataset_name = cfg["nnunet"]["dataset_name"]
        nnunet_raw = Path(cfg["nnunet"]["raw"])
        build_nnunet_raw(
            cases,
            train_txt=data_dir / "train.txt",
            test_txt=data_dir / "test.txt",
            nnunet_raw_dir=nnunet_raw,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
        )

    if not any([args.fix_cad_headers, args.update_txts, args.build_nnunet]):
        print("[INFO] No action flags specified. Use --help to see options.")


if __name__ == "__main__":
    main()
