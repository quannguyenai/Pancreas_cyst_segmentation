"""approach_d/prepare_dataset.py — Build nnUNet v1 Task001_PancreasCyst dataset.

Reads the predefined splits from data/train.txt, data/val.txt, data/test.txt
and creates symlinks + metadata files in the nnUNet v1 directory layout:

    nnunet_v1/raw/nnUNet_raw_data/Task001_PancreasCyst/
        imagesTr/  — train + val cases (284 total)
        labelsTr/  — train + val masks
        imagesTs/  — test cases (74 total)
        dataset.json
    nnunet_v1/preprocessed/Task001_PancreasCyst/
        splits_final.pkl  — predefined fold 0: 247 train / 37 val

Usage
-----
python approach_d/prepare_dataset.py --config configs/paths.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs import load_config


DATASET_JSON_V1 = {
    "name": "PancreasCyst",
    "description": "Pancreatic cyst segmentation, multi-institutional T2 MRI",
    "reference": "",
    "licence": "",
    "release": "1.0",
    "modality": {"0": "MRI"},
    "labels": {"0": "background", "1": "cyst"},
    "numTraining": 0,
    "numTest": 0,
    "training": [],
    "test": [],
}


def _read_stems_and_paths(txt: Path) -> list[tuple[str, Path, Path | None]]:
    """Return [(stem, img_path, mask_path_or_None), ...]."""
    rows = []
    for line in txt.read_text().splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        img = Path(parts[0])
        mask = Path(parts[1]) if len(parts) > 1 else None
        stem = img.name.replace(".nii.gz", "")
        # Strip nnUNet channel suffix if the source file already has _0000
        if len(stem) > 5 and stem[-5] == "_" and stem[-4:].isdigit():
            stem = stem[:-5]
        rows.append((stem, img, mask))
    return rows


def build_v1_dataset(
    repo_root: Path,
    raw_data_base: Path,
    preprocessed_base: Path,
    task_id: int = 1,
    task_name: str = "PancreasCyst",
) -> None:
    task_folder_name = f"Task{task_id:03d}_{task_name}"

    task_dir   = raw_data_base / "nnUNet_raw_data" / task_folder_name
    images_tr  = task_dir / "imagesTr"
    labels_tr  = task_dir / "labelsTr"
    images_ts  = task_dir / "imagesTs"
    preproc_dir = preprocessed_base / task_folder_name

    for d in (images_tr, labels_tr, images_ts, preproc_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Remove stale symlinks / files from previous runs to avoid nnUNet scanning them
    for d in (images_tr, labels_tr, images_ts):
        for f in d.iterdir():
            if f.is_symlink() or f.is_file():
                f.unlink()

    train_rows = _read_stems_and_paths(repo_root / "data/train.txt")
    val_rows   = _read_stems_and_paths(repo_root / "data/val.txt")
    test_rows  = _read_stems_and_paths(repo_root / "data/test.txt")

    train_stems = [s for s, _, _ in train_rows]
    val_stems   = [s for s, _, _ in val_rows]

    # imagesTr = train + val (both needed so nnUNet v1 can access val during training)
    imagestr_rows = train_rows + val_rows
    n_train = 0
    training_entries = []

    for stem, img_path, mask_path in imagestr_rows:
        dst_img  = images_tr / f"{stem}_0000.nii.gz"
        dst_mask = labels_tr / f"{stem}.nii.gz"

        for dst, src in [(dst_img, img_path), (dst_mask, mask_path)]:
            if src is None:
                continue
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src.resolve())

        training_entries.append({
            "image": f"./imagesTr/{stem}_0000.nii.gz",
            "label": f"./labelsTr/{stem}.nii.gz",
        })
        n_train += 1

    test_entries = []
    for stem, img_path, _ in test_rows:
        dst_img = images_ts / f"{stem}_0000.nii.gz"
        if dst_img.exists() or dst_img.is_symlink():
            dst_img.unlink()
        dst_img.symlink_to(img_path.resolve())
        test_entries.append(f"./imagesTs/{stem}_0000.nii.gz")

    # Write dataset.json (v1 format)
    ds_json = {**DATASET_JSON_V1}
    ds_json["numTraining"] = n_train
    ds_json["numTest"]     = len(test_entries)
    ds_json["training"]    = training_entries
    ds_json["test"]        = test_entries
    (task_dir / "dataset.json").write_text(json.dumps(ds_json, indent=4) + "\n")

    # Write splits_final.pkl with predefined 247/37 fold 0
    splits = [{"train": train_stems, "val": val_stems}]
    with open(preproc_dir / "splits_final.pkl", "wb") as f:
        pickle.dump(splits, f)

    print(f"[OK] Built {task_folder_name}")
    print(f"     imagesTr: {n_train} cases (train={len(train_stems)}, val={len(val_stems)})")
    print(f"     imagesTs: {len(test_entries)} cases")
    print(f"     splits_final.pkl: fold 0 — train={len(train_stems)}, val={len(val_stems)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build nnUNet v1 dataset for approach_d (nnTransUNetTrainerV2).",
    )
    p.add_argument("--config", default="configs/paths.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)

    repo_root        = Path(args.config).parent.parent.resolve()
    raw_data_base    = Path(__file__).parent / "nnunet_v1" / "raw"
    preprocessed_base = Path(__file__).parent / "nnunet_v1" / "preprocessed"

    # Allow env var overrides (set by set_env.sh)
    if os.environ.get("nnUNet_raw_data_base"):
        raw_data_base = Path(os.environ["nnUNet_raw_data_base"])
    if os.environ.get("nnUNet_preprocessed"):
        preprocessed_base = Path(os.environ["nnUNet_preprocessed"])

    build_v1_dataset(repo_root, raw_data_base, preprocessed_base)


if __name__ == "__main__":
    main()
