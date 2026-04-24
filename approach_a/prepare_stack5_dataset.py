"""prepare_stack5_dataset.py — Build Dataset011_PancreasCyst25D.

Approach A2.5D: stack-as-channels 2D. For each MRI case, emit 5 z-shifted
copies of the volume as channels _0000..._0004. At training time, nnU-Net
v2's built-in `2d` config samples one 2D slice at z0 across all channels,
yielding a 5-channel input [V[z0-2], V[z0-1], V[z0], V[z0+1], V[z0+2]].
The label at z0 is unchanged (middle-slice supervision).

Boundaries use replicate padding (first/last slice repeated).

Reads the same train/val/test .txt splits as Dataset001, so case assignment
is identical — only the input representation differs.

Usage
-----
python approach_a/prepare_stack5_dataset.py --config configs/paths.yaml
python approach_a/prepare_stack5_dataset.py --config configs/paths.yaml --dataset-id 11 --window 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs import load_config


DATASET_JSON_TEMPLATE = {
    "labels": {"background": 0, "cyst": 1},
    "file_ending": ".nii.gz",
    "overwrite_image_reader_writer": "SimpleITKIO",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build Dataset011_PancreasCyst25D (stack-as-channels 2.5D).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="configs/paths.yaml")
    p.add_argument("--dataset-id", type=int, default=11)
    p.add_argument("--dataset-name", default="PancreasCyst25D")
    p.add_argument(
        "--window", type=int, default=5, choices=[3, 5],
        help="Number of adjacent slices to stack as channels (odd).",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Overwrite existing channel files instead of skipping.",
    )
    return p.parse_args()


def read_cases(txt_path: Path) -> list[tuple[Path, Path]]:
    if not txt_path.exists():
        return []
    rows = txt_path.read_text().splitlines()[1:]
    return [
        (Path(r.split(",")[0]), Path(r.split(",")[1]))
        for r in rows if r.strip()
    ]


def shift_replicate(arr: np.ndarray, shift: int) -> np.ndarray:
    """Return out where out[z] = arr[clamp(z + shift, 0, Z-1)].

    arr is (Z, Y, X) as returned by sitk.GetArrayFromImage.
    """
    Z = arr.shape[0]
    idx = np.clip(np.arange(Z) + shift, 0, Z - 1)
    return arr[idx]


def write_shifted_channels(
    img_path: Path,
    out_dir: Path,
    stem: str,
    shifts: list[int],
    force: bool,
) -> None:
    """Read img_path, write len(shifts) channels into out_dir/{stem}_000C.nii.gz."""
    targets = [out_dir / f"{stem}_{c:04d}.nii.gz" for c in range(len(shifts))]
    if not force and all(t.exists() for t in targets):
        return

    img = sitk.ReadImage(str(img_path))
    arr = sitk.GetArrayFromImage(img)  # (Z, Y, X)

    for ch, shift in enumerate(shifts):
        target = targets[ch]
        if target.exists() and not force:
            continue
        shifted_arr = shift_replicate(arr, shift) if shift != 0 else arr
        out_img = sitk.GetImageFromArray(shifted_arr)
        out_img.CopyInformation(img)
        sitk.WriteImage(out_img, str(target), useCompression=True)


def symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.window % 2 == 0:
        raise SystemExit(f"--window must be odd, got {args.window}")

    half = args.window // 2
    shifts = list(range(-half, half + 1))  # e.g. [-2,-1,0,1,2] for window=5

    data_cfg = cfg["data"]
    train_cases = read_cases(Path(data_cfg["train_txt"]))
    val_cases   = read_cases(Path(data_cfg["val_txt"]))
    test_cases  = read_cases(Path(data_cfg["test_txt"]))

    imagestr_cases = train_cases + val_cases  # nnU-Net needs both in imagesTr

    nnunet_raw = Path(cfg["nnunet"]["raw"])
    dataset_folder = nnunet_raw / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    images_tr = dataset_folder / "imagesTr"
    labels_tr = dataset_folder / "labelsTr"
    images_ts = dataset_folder / "imagesTs"
    for d in (images_tr, labels_tr, images_ts):
        d.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Building {dataset_folder.name}")
    print(f"       window={args.window}  shifts={shifts}")
    print(f"       {len(imagestr_cases)} imagesTr cases, {len(test_cases)} imagesTs cases")

    # Train + val
    for i, (img_path, mask_path) in enumerate(imagestr_cases, 1):
        stem = img_path.name.replace(".nii.gz", "")
        write_shifted_channels(img_path, images_tr, stem, shifts, args.force)
        symlink(mask_path, labels_tr / f"{stem}.nii.gz")
        if i % 25 == 0 or i == len(imagestr_cases):
            print(f"       imagesTr: {i}/{len(imagestr_cases)}")

    # Test
    for i, (img_path, _mask_path) in enumerate(test_cases, 1):
        stem = img_path.name.replace(".nii.gz", "")
        write_shifted_channels(img_path, images_ts, stem, shifts, args.force)
        if i % 25 == 0 or i == len(test_cases):
            print(f"       imagesTs: {i}/{len(test_cases)}")

    ds_json = {
        **DATASET_JSON_TEMPLATE,
        "name": args.dataset_name,
        "description": (
            f"Stack-as-channels 2.5D view of PancreasCyst (MRI, T1/T2). Each "
            f"case's volume is written as {args.window} z-shifted copies "
            f"(shifts={shifts}); channel {half} is the unshifted centre. Train "
            f"the built-in `2d` configuration — each 2D sample becomes a "
            f"{args.window}-channel input."
        ),
        "channel_names": {str(c): "MRI" for c in range(args.window)},
        "numTraining": len(imagestr_cases),
    }
    (dataset_folder / "dataset.json").write_text(
        json.dumps(ds_json, indent=4) + "\n"
    )
    print(f"[INFO] Wrote dataset.json ({args.window} MRI channels).")


if __name__ == "__main__":
    main()
