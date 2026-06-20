"""train.py — Unified training script for comparison baseline models.

Supports 3-D V-Net / U-Net and 2-D U-Net modes via --mode flag.
Replaces baseline/3D-VNet/train3d.py and baseline/2D-UNet/train2d.py.

Usage
-----
# 3-D V-Net (supervised baseline):
python comparison/train.py --config configs/paths.yaml --mode 3d --model vnet

# 2-D U-Net:
python comparison/train.py --config configs/paths.yaml --mode 2d --model unet_2d

# Semi-supervised with BCP:
python comparison/train.py --config configs/paths.yaml --mode 3d --model vnet \\
    --labelnum 50 --consistency 1.0
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs import load_config
from comparison.networks.net_factory import net_factory
from comparison.dataloaders.dataset import (
    Cyst, Cyst2D, CenterCrop, RandomCrop, RandomRotFlip, Normalize, ToTensor,
    TwoStreamBatchSampler,
)
from comparison.utils import losses, ramps


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train comparison baseline segmentation models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="configs/paths.yaml")
    p.add_argument("--mode", choices=["2d", "3d"], default="3d",
                   help="2d = 2D U-Net on axial slices; 3d = 3D volumetric model")
    p.add_argument("--model", default="vnet",
                   help="Network type: vnet | unet_2d | unet_3d | unetr")
    p.add_argument("--exp", default="baseline",
                   help="Experiment name (used for checkpoint subdirectory)")
    p.add_argument("--fold", type=int, default=None,
                   help="5-fold CV fold index. When set, train/val come from "
                        "comparison/splits/fold{fold}_{train,val}.txt and "
                        "checkpoints go to <checkpoint_dir>/<exp>/fold{fold}/.")
    p.add_argument("--splits-dir", default=None,
                   help="Directory holding fold{k}_{train,val}.txt "
                        "(default: comparison/splits next to this script).")
    p.add_argument("--max-epoch", type=int, default=80)
    p.add_argument("--batchsize", type=int, default=4)
    p.add_argument("--base-lr", type=float, default=0.01)
    p.add_argument("--labelnum", type=int, default=None,
                   help="Use only first N labelled samples (semi-supervised)")
    p.add_argument("--gpu", default="0")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--patch-size", type=int, nargs="+", default=None,
                   help="Override patch size from config")
    # Semi-supervised BCP settings
    p.add_argument("--consistency", type=float, default=0.0)
    p.add_argument("--consistency-rampup", type=float, default=40.0)
    p.add_argument("--u-weight", type=float, default=0.5)
    p.add_argument("--resume", default=None,
                   help="Path to checkpoint_latest.pth to resume interrupted training")
    return p.parse_args()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def get_current_consistency_weight(epoch: int, rampup: float) -> float:
    return ramps.sigmoid_rampup(epoch, rampup)


def build_transform_3d(patch_size):
    return transforms.Compose([
        Normalize(),
        CenterCrop(patch_size),
        RandomRotFlip(),
        ToTensor(),
    ])


def build_transform_2d(patch_size):
    from comparison.dataloaders.dataset import RandomGenerator
    return RandomGenerator(patch_size[:2])


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_seed(args.seed)

    # Resolve paths
    if args.fold is not None:
        splits_dir = Path(args.splits_dir) if args.splits_dir else (
            Path(__file__).parent / "splits"
        )
        train_txt = splits_dir / f"fold{args.fold}_train.txt"
        val_txt   = splits_dir / f"fold{args.fold}_val.txt"
        if not train_txt.exists() or not val_txt.exists():
            raise SystemExit(
                f"Fold splits not found: {train_txt} / {val_txt}. "
                "Run comparison/make_cv_splits.py first."
            )
        ckpt_dir = Path(cfg["comparison"]["checkpoint_dir"]) / args.exp / f"fold{args.fold}"
    else:
        train_txt = Path(cfg["data"]["train_txt"])
        val_txt   = Path(cfg["data"]["val_txt"])
        ckpt_dir  = Path(cfg["comparison"]["checkpoint_dir"]) / args.exp
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    patch_size = args.patch_size or (
        cfg["comparison"]["patch_size_3d"] if args.mode == "3d"
        else cfg["comparison"]["patch_size_2d"]
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(ckpt_dir / "train.log"),
            logging.StreamHandler(),
        ],
    )
    logging.info(f"Experiment: {args.exp} | mode: {args.mode} | model: {args.model} "
                 f"| fold: {args.fold}")
    logging.info(f"train_txt: {train_txt}  val_txt: {val_txt}")
    logging.info(f"Patch size: {patch_size} | batch: {args.batchsize} | lr: {args.base_lr}")

    # Build datasets
    if args.mode == "3d":
        tf_train = build_transform_3d(patch_size)
        train_ds = Cyst(txt_path=train_txt, num=args.labelnum, transform=tf_train)
        val_ds   = Cyst(txt_path=val_txt,   transform=build_transform_3d(patch_size))
    else:
        tf_train = build_transform_2d(patch_size)
        train_ds = Cyst2D(txt_path=train_txt, split="train", transform=tf_train)
        val_ds   = Cyst2D(txt_path=val_txt,   split="val")

    train_loader = DataLoader(train_ds, batch_size=args.batchsize,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1,
                              shuffle=False, num_workers=2)

    # Build model
    model = net_factory(args.model, in_chns=1, class_num=2).cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr,
                          momentum=0.9, weight_decay=1e-4)

    ce_loss   = nn.CrossEntropyLoss()
    dice_loss = losses.DiceLoss(2)

    best_dice   = 0.0
    start_epoch = 0

    # Resume from checkpoint if requested or if latest checkpoint exists
    resume_path = args.resume or str(ckpt_dir / "checkpoint_latest.pth")
    if Path(resume_path).exists():
        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_dice   = ckpt.get("best_dice", 0.0)
        logging.info(f"Resumed from {resume_path} (epoch {start_epoch}, best_dice={best_dice:.4f})")

    for epoch in range(start_epoch, args.max_epoch):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.max_epoch}"):
            images = batch["image"].cuda().float()
            labels = batch["label"].cuda().long()

            outputs = model(images)
            outputs = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            loss_ce   = ce_loss(outputs, labels)
            # DiceLoss._one_hot_encoder cats along dim=1, so the target needs an
            # explicit channel dim [B,1,...]; CE keeps the [B,...] target.
            loss_dice = dice_loss(outputs, labels.unsqueeze(1), softmax=True)
            loss = 0.5 * (loss_ce + loss_dice)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1}: train_loss={avg_loss:.4f}")

        # Adjust LR (polynomial decay)
        lr = args.base_lr * (1 - epoch / args.max_epoch) ** 0.9
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Save latest checkpoint every epoch for crash recovery
        torch.save({
            "epoch":      epoch,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "best_dice":  best_dice,
        }, ckpt_dir / "checkpoint_latest.pth")

        # Validation every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            dice_scores = []
            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].cuda().float()
                    labels = batch["label"].numpy()
                    logits = model(images)
                    logits = logits[0] if isinstance(logits, (tuple, list)) else logits
                    preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
                    preds = preds.cpu().numpy()
                    for pred, gt in zip(preds, labels):
                        if gt.sum() > 0:
                            intersection = (pred * gt).sum()
                            dice = 2 * intersection / (pred.sum() + gt.sum() + 1e-8)
                            dice_scores.append(float(dice))

            val_dice = np.mean(dice_scores) if dice_scores else 0.0
            logging.info(f"Epoch {epoch+1}: val_dice={val_dice:.4f}")

            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(model.state_dict(), ckpt_dir / "best_model.pth")
                logging.info(f"  → New best checkpoint saved (dice={best_dice:.4f})")

    torch.save(model.state_dict(), ckpt_dir / "final_model.pth")
    logging.info(f"Training complete. Best val Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
