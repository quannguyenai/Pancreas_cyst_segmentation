"""Build split files and nnUNet raw dataset for cropped pancreas ROIs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs import load_config


DATASET_JSON_TEMPLATE = {
    "channel_names": {"0": "MRI"},
    "labels": {"background": 0, "cyst": 1},
    "numTraining": 0,
    "file_ending": ".nii.gz",
    "overwrite_image_reader_writer": "SimpleITKIO",
}

# Per-variant nnUNet dataset settings. ``priorchan`` adds a 2nd input channel
# (the pancreas distance map) symlinked as ``_0001``.
VARIANTS = {
    "default":   {"channel_names": {"0": "MRI"}, "with_prior": False,
                  "id_key": "nnunet_dataset_id",   "name_key": "nnunet_dataset_name"},
    "fixedbox":  {"channel_names": {"0": "MRI"}, "with_prior": False,
                  "id_key": "fixedbox_dataset_id", "name_key": "fixedbox_dataset_name"},
    "priorchan": {"channel_names": {"0": "MRI", "1": "pancreas"}, "with_prior": True,
                  "id_key": "priorchan_dataset_id", "name_key": "priorchan_dataset_name"},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create cropped split txt files and optional nnUNet dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="configs/paths.yaml")
    p.add_argument(
        "--variant", default="default", choices=list(VARIANTS),
        help="Dataset variant: default/fixedbox (1-ch) or priorchan (2-ch + pancreas prior).",
    )
    p.add_argument(
        "--skip-nnunet", action="store_true",
        help="Only write cropped split txt files, do not build nnUNet_raw dataset.",
    )
    return p.parse_args()


def read_rows(txt_path: Path) -> list[list[str]]:
    with open(txt_path) as f:
        rows = [line.strip().split(",") for line in f.readlines()[1:] if line.strip()]
    return rows


def write_cropped_split(
    src_txt: Path,
    dst_txt: Path,
    cropped_images: Path,
    cropped_masks: Path,
    require_masks: bool,
) -> list[tuple[str, Path, Path | None]]:
    rows = read_rows(src_txt)
    mapped: list[tuple[str, Path, Path | None]] = []
    out_lines = ["image_path,mask_path"]

    for parts in rows:
        image_path = Path(parts[0])
        stem = image_path.name.replace(".nii.gz", "")
        cropped_img = cropped_images / f"{stem}.nii.gz"
        if not cropped_img.exists():
            raise FileNotFoundError(f"Missing cropped image for {stem}: {cropped_img}")

        cropped_mask: Path | None = None
        if len(parts) > 1 and parts[1]:
            candidate = cropped_masks / f"{stem}.nii.gz"
            if candidate.exists():
                cropped_mask = candidate
            elif require_masks:
                raise FileNotFoundError(f"Missing cropped mask for {stem}: {candidate}")

        if cropped_mask is None:
            out_lines.append(str(cropped_img))
        else:
            out_lines.append(f"{cropped_img},{cropped_mask}")
        mapped.append((stem, cropped_img, cropped_mask))

    dst_txt.parent.mkdir(parents=True, exist_ok=True)
    dst_txt.write_text("\n".join(out_lines) + "\n")
    return mapped


def _symlink(dst: Path, src: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def build_nnunet_raw_dataset(
    train_rows: list[tuple[str, Path, Path | None]],
    val_rows: list[tuple[str, Path, Path | None]],
    test_rows: list[tuple[str, Path, Path | None]],
    dataset_dir: Path,
    dataset_name: str,
    channel_names: dict[str, str],
    prior_dir: Path | None,
) -> None:
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_ts = dataset_dir / "imagesTs"
    for directory in (images_tr, labels_tr, images_ts):
        directory.mkdir(parents=True, exist_ok=True)

    def prior_for(stem: str) -> Path:
        prior = prior_dir / f"{stem}.nii.gz"
        if not prior.exists():
            raise FileNotFoundError(
                f"Missing pancreas-prior crop for {stem}: {prior} "
                f"(re-run crop_to_pancreas.py --emit-distance-channel)"
            )
        return prior

    num_training = 0
    for stem, image_path, mask_path in [*train_rows, *val_rows]:
        if mask_path is None:
            raise ValueError(f"Training case {stem} is missing a cropped mask")

        _symlink(images_tr / f"{stem}_0000.nii.gz", image_path)
        _symlink(labels_tr / f"{stem}.nii.gz", mask_path)
        if prior_dir is not None:
            _symlink(images_tr / f"{stem}_0001.nii.gz", prior_for(stem))
        num_training += 1

    for stem, image_path, _mask_path in test_rows:
        _symlink(images_ts / f"{stem}_0000.nii.gz", image_path)
        if prior_dir is not None:
            _symlink(images_ts / f"{stem}_0001.nii.gz", prior_for(stem))

    dataset_json = {**DATASET_JSON_TEMPLATE}
    dataset_json["channel_names"] = channel_names
    dataset_json["name"] = dataset_name
    dataset_json["numTraining"] = num_training
    (dataset_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=4) + "\n")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    cropped_cfg = cfg["approach_b"]
    cropped_images = Path(cropped_cfg["cropped_images"])
    cropped_masks = Path(cropped_cfg["cropped_masks"])

    train_rows = write_cropped_split(
        src_txt=Path(cfg["data"]["train_txt"]),
        dst_txt=Path(cropped_cfg["cropped_train_txt"]),
        cropped_images=cropped_images,
        cropped_masks=cropped_masks,
        require_masks=True,
    )
    val_rows = write_cropped_split(
        src_txt=Path(cfg["data"]["val_txt"]),
        dst_txt=Path(cropped_cfg["cropped_val_txt"]),
        cropped_images=cropped_images,
        cropped_masks=cropped_masks,
        require_masks=True,
    )
    test_rows = write_cropped_split(
        src_txt=Path(cfg["data"]["test_txt"]),
        dst_txt=Path(cropped_cfg["cropped_test_txt"]),
        cropped_images=cropped_images,
        cropped_masks=cropped_masks,
        require_masks=False,
    )

    print(f"[INFO] Wrote cropped train split: {cropped_cfg['cropped_train_txt']}")
    print(f"[INFO] Wrote cropped val split:   {cropped_cfg['cropped_val_txt']}")
    print(f"[INFO] Wrote cropped test split:  {cropped_cfg['cropped_test_txt']}")

    if args.skip_nnunet:
        return

    variant = VARIANTS[args.variant]
    dataset_id = int(cropped_cfg[variant["id_key"]])
    dataset_name = cropped_cfg[variant["name_key"]]
    dataset_dir = Path(cfg["nnunet"]["raw"]) / f"Dataset{dataset_id:03d}_{dataset_name}"
    prior_dir = Path(cropped_cfg["cropped_prior"]) if variant["with_prior"] else None

    build_nnunet_raw_dataset(
        train_rows=train_rows,
        val_rows=val_rows,
        test_rows=test_rows,
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        channel_names=variant["channel_names"],
        prior_dir=prior_dir,
    )
    print(f"[INFO] Built cropped nnUNet dataset ({args.variant}): {dataset_dir}")


if __name__ == "__main__":
    main()
