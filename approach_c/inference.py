"""inference.py — Run inference with a fine-tuned approach_c model.

Usage
-----
python approach_c/inference.py \\
    --config configs/paths.yaml \\
    --checkpoint approach_c/checkpoints/best_model.pth \\
    --split test \\
    --output approach_c/predictions
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs import load_config
from approach_c.finetune_trainer import build_model
from comparison.utils.metrics import calculate_metric_percase

try:
    from monai.inferers import sliding_window_inference
    _MONAI_AVAILABLE = True
except ImportError:
    _MONAI_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inference with fine-tuned approach_c model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="configs/paths.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--output", default=None)
    p.add_argument("--gpu", default="0")
    return p.parse_args()


def preprocess_image(image: np.ndarray) -> torch.Tensor:
    """Clip to CT soft-tissue window [-100, 400 HU] and normalise to [0, 1]."""
    image = np.clip(image, -100, 400)
    image = (image - (-100)) / (400 - (-100))
    return torch.from_numpy(image.astype(np.float32)).unsqueeze(0).unsqueeze(0)


def main() -> None:
    import os
    args = parse_args()
    cfg  = load_config(args.config)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if not _MONAI_AVAILABLE:
        raise RuntimeError("MONAI is required. Install: pip install monai")

    patch_size = cfg["approach_c"]["patch_size"]
    output_dir = Path(args.output or cfg["approach_c"]["predictions"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = build_model(cfg, pretrained_weights=None)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval().cuda()

    # Load split
    split_map = {
        "train": cfg["data"]["train_txt"],
        "val":   cfg["data"]["val_txt"],
        "test":  cfg["data"]["test_txt"],
    }
    with open(split_map[args.split]) as f:
        lines = f.readlines()[1:]
    cases = [line.strip().split(",") for line in lines if line.strip()]

    metrics = []
    with torch.no_grad():
        for parts in cases:
            img_path  = parts[0]
            mask_path = parts[1] if len(parts) > 1 else None

            img_nib  = nib.load(img_path)
            image    = img_nib.get_fdata().astype(np.float32)
            inp      = preprocess_image(image).cuda()

            pred_tensor = sliding_window_inference(
                inp, roi_size=patch_size, sw_batch_size=2,
                predictor=model, overlap=0.5,
            )
            pred = torch.argmax(torch.softmax(pred_tensor, dim=1), dim=1)
            pred = pred.squeeze().cpu().numpy().astype(np.uint8)

            # Save prediction
            stem = Path(img_path).name.replace(".nii.gz", "")
            out_nib = nib.Nifti1Image(pred, affine=img_nib.affine,
                                       header=img_nib.header)
            nib.save(out_nib, str(output_dir / f"{stem}.nii.gz"))

            # Compute metrics if GT available
            if mask_path and Path(mask_path).exists():
                gt = nib.load(mask_path).get_fdata().astype(np.uint8)
                if gt.sum() > 0:
                    try:
                        m = calculate_metric_percase(pred > 0, gt > 0)
                        # m = (dc, jc, hd95, asd)
                        metrics.append({"case": stem, "dice": m[0], "hd95": m[2], "asd": m[3]})
                        print(f"  {stem}: Dice={m[0]:.4f}  HD95={m[2]:.2f}  ASD={m[3]:.2f}")
                    except Exception:
                        pass

    if metrics:
        dice_vals = [m["dice"] for m in metrics]
        hd95_vals = [m["hd95"] for m in metrics]
        asd_vals  = [m["asd"]  for m in metrics]
        print(f"\nDice: {np.mean(dice_vals):.4f} ± {np.std(dice_vals):.4f}")
        print(f"HD95: {np.mean(hd95_vals):.2f} ± {np.std(hd95_vals):.2f}")
        print(f"ASD:  {np.mean(asd_vals):.2f} ± {np.std(asd_vals):.2f}")

    print(f"\nPredictions saved to: {output_dir}")


if __name__ == "__main__":
    main()
