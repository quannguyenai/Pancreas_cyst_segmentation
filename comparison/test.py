"""test.py — Unified inference / evaluation script for comparison baselines.

Usage
-----
python comparison/test.py \\
    --config configs/paths.yaml \\
    --mode 3d \\
    --model vnet \\
    --checkpoint comparison/checkpoints/baseline/best_model.pth \\
    --output comparison/predictions/vnet
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs import load_config
from comparison.networks.net_factory import net_factory
from comparison.utils.test_3d_patch import test_single_case
from comparison.utils.metrics import calculate_metric_percase


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run inference with a trained comparison baseline model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="configs/paths.yaml")
    p.add_argument("--mode", choices=["2d", "3d"], default="3d")
    p.add_argument("--model", default="vnet")
    p.add_argument("--checkpoint", default=None,
                   help="Path to a single .pth checkpoint.")
    p.add_argument("--checkpoints", nargs="+", default=None,
                   help="Multiple .pth checkpoints (e.g. the 5 fold best_model.pth). "
                        "When >1, predictions are the softmax ensemble of all folds "
                        "(matches the nnU-Net 5-fold ensemble protocol).")
    p.add_argument("--split", default="test",
                   choices=["train", "val", "test"],
                   help="Which split CSV to run inference on")
    p.add_argument("--output", default=None,
                   help="Directory to save predicted NIfTI masks")
    p.add_argument("--per-fold-csv", default=None,
                   help="Optional path to write each fold's standalone test Dice "
                        "(only meaningful when ensembling).")
    p.add_argument("--gpu", default="0")
    p.add_argument("--patch-size", type=int, nargs="+", default=None)
    p.add_argument("--stride", type=int, nargs="+", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    split_map = {
        "train": cfg["data"]["train_txt"],
        "val":   cfg["data"]["val_txt"],
        "test":  cfg["data"]["test_txt"],
    }
    txt_path = Path(split_map[args.split])

    patch_size = args.patch_size or (
        cfg["comparison"]["patch_size_3d"] if args.mode == "3d"
        else cfg["comparison"]["patch_size_2d"]
    )
    stride = args.stride or (
        cfg["comparison"]["stride_3d"] if args.mode == "3d"
        else cfg["comparison"]["stride_2d"]
    )

    output_dir = Path(args.output) if args.output else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve checkpoint list (single model or 5-fold ensemble).
    ckpt_paths = args.checkpoints or ([args.checkpoint] if args.checkpoint else None)
    if not ckpt_paths:
        raise SystemExit("Provide --checkpoint or --checkpoints.")
    models = []
    for cp in ckpt_paths:
        m = net_factory(args.model, in_chns=1, class_num=2)
        m.load_state_dict(torch.load(cp, map_location="cpu"))
        m.eval().cuda()
        models.append(m)
    ensemble = len(models) > 1
    logging.info(f"Loaded {len(models)} model(s) "
                 f"({'ensemble' if ensemble else 'single'}): {ckpt_paths}")

    with open(txt_path) as f:
        lines = f.readlines()[1:]
    cases = [line.strip().split(",") for line in lines if line.strip()]

    def fg_prob(logit_map: np.ndarray) -> np.ndarray:
        """Stable softmax foreground prob from averaged logits [2,W,H,D]."""
        mx = logit_map.max(axis=0, keepdims=True)
        e = np.exp(logit_map - mx)
        return (e[1] / e.sum(axis=0))

    metrics: list[dict] = []
    per_fold_rows: list[dict] = []
    for image_path, mask_path in cases:
        image_nib = nib.load(image_path)
        image     = image_nib.get_fdata().astype(np.float32)
        label     = nib.load(mask_path).get_fdata().astype(np.uint8)
        stem      = Path(image_path).name.replace(".nii.gz", "")

        # Foreground normalisation (same as training)
        fg = image > 0
        if fg.any():
            image = (image - image[fg].mean()) / (image[fg].std() + 1e-8)

        prob_sum = None
        for fi, model in enumerate(models):
            label_map, logit_map = test_single_case(
                model, image, stride_xy=stride[0], stride_z=stride[-1],
                patch_size=patch_size, num_classes=2,
            )
            p = fg_prob(logit_map)
            prob_sum = p if prob_sum is None else prob_sum + p
            if ensemble and label.sum() > 0:
                d = calculate_metric_percase(label_map > 0, label > 0)[0]
                per_fold_rows.append({"case": stem, "fold": fi, "dice": d})

        prediction = (prob_sum / len(models) > 0.5).astype(np.uint8)

        if label.sum() > 0:
            # masks encode foreground inconsistently (1 / 255 / 65535) -> binarize >0
            m = calculate_metric_percase(prediction > 0, label > 0)
            # m = (dc, jc, hd95, asd)
            metrics.append({"case": image_path, "dice": m[0], "hd95": m[2], "asd": m[3]})
            logging.info(f"{stem}: Dice={m[0]:.4f}  HD95={m[2]:.2f}  ASD={m[3]:.2f}")

        if output_dir:
            pred_nib = nib.Nifti1Image(
                prediction.astype(np.uint8),
                affine=image_nib.affine,
                header=image_nib.header,
            )
            nib.save(pred_nib, output_dir / f"{stem}.nii.gz")

    if per_fold_rows and args.per_fold_csv:
        import pandas as pd
        pf = Path(args.per_fold_csv)
        pf.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(per_fold_rows).to_csv(pf, index=False, float_format="%.4f")
        logging.info(f"Per-fold test Dice written to {pf}")

    if metrics:
        dice_vals = [m["dice"] for m in metrics]
        hd95_vals = [m["hd95"] for m in metrics]
        asd_vals  = [m["asd"]  for m in metrics]
        print(f"\n{'='*50}")
        tag = "5-fold ensemble" if ensemble else "single model"
        print(f"Results on {args.split} split ({len(metrics)} cases with GT, {tag}):")
        print(f"  Dice: {np.mean(dice_vals):.4f} ± {np.std(dice_vals):.4f}")
        print(f"  HD95: {np.mean(hd95_vals):.2f} ± {np.std(hd95_vals):.2f}")
        print(f"  ASD:  {np.mean(asd_vals):.2f} ± {np.std(asd_vals):.2f}")
        print(f"  (canonical comparable metrics: run comparison/eval_from_txt.py "
              f"on {output_dir})")
        print(f"{'='*50}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
