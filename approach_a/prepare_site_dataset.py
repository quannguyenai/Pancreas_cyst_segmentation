"""prepare_site_dataset.py — Build a per-institution nnUNet dataset.

Called by approach_a/train_per_modality.sh for each institution.
Filters the master train/val/test CSVs by institution prefix (e.g. "AHN"),
then creates a minimal nnUNet_raw/DatasetXXX_SITE/ with symlinks.

Usage
-----
python approach_a/prepare_site_dataset.py \\
    --config configs/paths.yaml \\
    --site AHN \\
    --dataset-id 2
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs import load_config

DATASET_JSON_TEMPLATE = {
    "channel_names": {"0": "CT"},
    "labels": {"background": 0, "cyst": 1},
    "numTraining": 0,
    "file_ending": ".nii.gz",
    "overwrite_image_reader_writer": "SimpleITKIO",
}

_INST_PATTERN = re.compile(r"^([A-Za-z]+?)(\d+)$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a per-institution nnUNet dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="configs/paths.yaml")
    p.add_argument("--site", required=True,
                   help="Institution prefix, e.g. AHN, CAD, EMC ...")
    p.add_argument("--dataset-id", type=int, required=True,
                   help="nnUNet dataset ID (e.g. 2 → Dataset002_AHN)")
    return p.parse_args()


def cases_from_txt(txt_path: Path) -> list[tuple[str, str]]:
    """Return [(image_path, mask_path), ...] from a split CSV."""
    with open(txt_path) as f:
        lines = f.readlines()[1:]
    return [tuple(line.strip().split(",")) for line in lines if line.strip()]


def filter_by_site(cases: list[tuple[str, str]], site: str) -> list[tuple[str, str]]:
    """Keep only cases whose image stem starts with ``site`` (case-insensitive)."""
    prefix = site.upper()
    filtered = []
    for img_path, mask_path in cases:
        stem = Path(img_path).name.replace(".nii.gz", "")
        m = _INST_PATTERN.match(stem)
        if m and m.group(1).upper() == prefix:
            filtered.append((img_path, mask_path))
    return filtered


def symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    site = args.site.upper()

    data_dir = Path(cfg["root"]) / "data"
    nnunet_raw = Path(cfg["nnunet"]["raw"])

    # Load all splits
    all_cases: list[tuple[str, str]] = []
    for split_key in ("train_txt", "val_txt"):
        txt = Path(cfg["data"][split_key])
        all_cases.extend(cases_from_txt(txt))

    test_cases = cases_from_txt(Path(cfg["data"]["test_txt"]))

    site_train = filter_by_site(all_cases, site)
    site_test  = filter_by_site(test_cases, site)

    if not site_train:
        print(f"[WARN] No training cases found for site={site}. Skipping.")
        return

    dataset_name = f"Dataset{args.dataset_id:03d}_{site}"
    dataset_folder = nnunet_raw / dataset_name
    images_tr = dataset_folder / "imagesTr"
    labels_tr  = dataset_folder / "labelsTr"
    images_ts  = dataset_folder / "imagesTs"

    for d in (images_tr, labels_tr, images_ts):
        d.mkdir(parents=True, exist_ok=True)

    for img_path, mask_path in site_train:
        stem = Path(img_path).name.replace(".nii.gz", "")
        symlink(Path(img_path), images_tr / f"{stem}_0000.nii.gz")
        symlink(Path(mask_path), labels_tr / f"{stem}.nii.gz")

    for img_path, _ in site_test:
        stem = Path(img_path).name.replace(".nii.gz", "")
        symlink(Path(img_path), images_ts / f"{stem}_0000.nii.gz")

    ds_json = {**DATASET_JSON_TEMPLATE,
               "numTraining": len(site_train),
               "name": dataset_name}
    (dataset_folder / "dataset.json").write_text(
        json.dumps(ds_json, indent=4) + "\n"
    )

    print(f"[INFO] {dataset_name}: {len(site_train)} train, {len(site_test)} test cases.")


if __name__ == "__main__":
    main()
