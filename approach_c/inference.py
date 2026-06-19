"""inference.py — Sliding-window inference with fine-tuned PanSegNet.

No MONAI dependency.

Usage
-----
python approach_c/inference.py \\
    --config configs/paths.yaml \\
    --checkpoint approach_c/checkpoints/best_model.pth \\
    --split test \\
    --output approach_c/predictions/test
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from configs import load_config
from comparison.utils.metrics import calculate_metric_percase
from pansegnet import PanSegNet
from finetune_trainer import sliding_window_inference, _normalize


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config",     default="configs/paths.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split",      default="test", choices=["train", "val", "test"])
    p.add_argument("--output",     default=None)
    p.add_argument("--gpu",        default="0")
    p.add_argument("--overlap",    type=float, default=0.5)
    p.add_argument("--sw-batch",   type=int,   default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    patch_size = cfg["approach_c"]["patch_size"]
    output_dir = Path(args.output or cfg["approach_c"]["predictions"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    model = PanSegNet(input_channels=1, num_classes=2, patch_size=patch_size)
    ckpt  = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval().cuda()
    model.do_ds = False   # single output at inference

    # ── Load split CSV ────────────────────────────────────────────────────────
    split_map = {"train": cfg["data"]["train_txt"],
                 "val":   cfg["data"]["val_txt"],
                 "test":  cfg["data"]["test_txt"]}
    with open(split_map[args.split]) as f:
        lines = f.readlines()[1:]
    cases = [l.strip().split(",") for l in lines if l.strip()]

    metrics: list[dict] = []

    for parts in cases:
        img_path  = parts[0].strip()
        mask_path = parts[1].strip() if len(parts) > 1 else None

        img_nib = nib.load(img_path)
        image   = img_nib.get_fdata().astype(np.float32)
        image   = _normalize(image)
        volume  = torch.from_numpy(image[None, None]).cuda().float()  # [1,1,H,W,D]

        pred_logits = sliding_window_inference(
            model, volume, patch_size,
            overlap=args.overlap, sw_batch_size=args.sw_batch,
        )
        pred = torch.argmax(pred_logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        stem    = Path(img_path).name.replace(".nii.gz", "")
        out_nib = nib.Nifti1Image(pred, affine=img_nib.affine, header=img_nib.header)
        nib.save(out_nib, str(output_dir / f"{stem}.nii.gz"))

        if mask_path and Path(mask_path).exists():
            gt = nib.load(mask_path).get_fdata().astype(np.uint8)
            if gt.sum() > 0:
                try:
                    m = calculate_metric_percase(pred > 0, gt > 0)
                    metrics.append({"case": stem, "dice": m[0], "hd95": m[2], "asd": m[3]})
                    print(f"  {stem}  Dice={m[0]:.4f}  HD95={m[2]:.2f}  ASD={m[3]:.2f}")
                except Exception:
                    pass

    if metrics:
        d = [m["dice"] for m in metrics]
        h = [m["hd95"] for m in metrics]
        a = [m["asd"]  for m in metrics]
        print(f"\nDice : {np.mean(d):.4f} ± {np.std(d):.4f}")
        print(f"HD95 : {np.mean(h):.2f} ± {np.std(h):.2f}")
        print(f"ASD  : {np.mean(a):.2f} ± {np.std(a):.2f}")

    print(f"\nPredictions saved → {output_dir}")


if __name__ == "__main__":
    main()
