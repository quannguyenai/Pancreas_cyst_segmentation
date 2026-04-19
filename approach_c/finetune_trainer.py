"""finetune_trainer.py — Fine-tune a pretrained PanSegNet encoder for cyst segmentation.

Architecture: MONAI UNet backbone with pretrained encoder weights loaded from
PanSegNet. The final segmentation head is replaced for binary (background/cyst)
output. The encoder is frozen for ``warmup_freeze_epochs``, then all parameters
are unfrozen and trained end-to-end.

Training uses DiceCELoss, CosineAnnealingLR, and early stopping on validation Dice.
Sliding-window inference (MONAI) is used for validation and inference.

Usage
-----
# Training:
python approach_c/finetune_trainer.py --config configs/paths.yaml

# Resume from checkpoint:
python approach_c/finetune_trainer.py --config configs/paths.yaml \\
    --resume approach_c/checkpoints/epoch_10_dice_0.7234.pth

Prerequisites
-------------
* PanSegNet pretrained weights placed at:
    approach_c/pretrained/PanSegNet.pth
  (obtain from the original authors)
* MONAI >= 1.5.2, PyTorch >= 2.0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs import load_config
from comparison.dataloaders.dataset import Normalize, ToTensor, RandomCrop, CenterCrop
from comparison.utils.metrics import calculate_metric_percase

try:
    from monai.losses import DiceCELoss
    from monai.inferers import sliding_window_inference
    from monai.networks.nets import UNet
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
        Spacingd, ScaleIntensityRanged, RandCropByPosNegLabeld,
        RandFlipd, RandRotate90d, ToTensord,
    )
    from monai.data import CacheDataset, DataLoader as MonaiDataLoader
    _MONAI_AVAILABLE = True
except ImportError:
    _MONAI_AVAILABLE = False
    print("[WARN] MONAI not available; falling back to custom data pipeline.")

import nibabel as nib


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune PanSegNet encoder for pancreatic cyst segmentation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="configs/paths.yaml")
    p.add_argument("--resume", default=None,
                   help="Path to checkpoint to resume from.")
    p.add_argument("--freeze-epochs", type=int, default=None,
                   help="Override config warmup_freeze_epochs.")
    p.add_argument("--fold", type=int, default=0,
                   help="Fold index for cross-validation (0 = train on train.txt).")
    p.add_argument("--gpu", default="0")
    p.add_argument("--batch_size", type=int, default=None,
                   help="Override config batch_size.")
    p.add_argument("--num_workers", type=int, default=None,
                   help="DataLoader num_workers (default: 4).")
    p.add_argument("--sw_batch_size", type=int, default=2,
                   help="Sliding-window batch size during validation (default: 2).")
    p.add_argument("--cache_rate", type=float, default=0.0,
                   help="MONAI CacheDataset cache_rate (default: 0.0 = no caching).")
    return p.parse_args()


# ─── MONAI data pipeline ──────────────────────────────────────────────────────

def build_monai_dataloaders(cfg: dict, args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    """Build MONAI CacheDataset loaders from split CSV files."""
    patch_size   = cfg["approach_c"]["patch_size"]
    batch_size   = args.batch_size if args.batch_size is not None else int(cfg["approach_c"]["batch_size"])
    num_workers  = args.num_workers if args.num_workers is not None else 4
    cache_rate   = args.cache_rate

    def load_csv(txt_path: str) -> list[dict]:
        data = []
        with open(txt_path) as f:
            for line in f.readlines()[1:]:
                line = line.strip()
                if line:
                    img, mask = line.split(",")
                    data.append({"image": img.strip(), "label": mask.strip()})
        return data

    train_data = load_csv(cfg["data"]["train_txt"])
    val_data   = load_csv(cfg["data"]["val_txt"])

    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-100, a_max=400,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=1, neg=1,
            num_samples=4,
            allow_smaller=True,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.1, max_k=3),
        ToTensord(keys=["image", "label"]),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-100, a_max=400,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
        ToTensord(keys=["image", "label"]),
    ])

    train_ds = CacheDataset(train_data, transform=train_transforms,
                            cache_rate=cache_rate, num_workers=num_workers)
    val_ds   = CacheDataset(val_data,   transform=val_transforms,
                            cache_rate=cache_rate, num_workers=num_workers)

    train_loader = MonaiDataLoader(train_ds,
                                   batch_size=batch_size,
                                   shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = MonaiDataLoader(val_ds,
                                   batch_size=1,
                                   shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


# ─── Model ────────────────────────────────────────────────────────────────────

def build_model(cfg: dict, pretrained_weights: str | None) -> nn.Module:
    """Build MONAI UNet and optionally load pretrained encoder weights."""
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(32, 64, 128, 256, 320),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="instance",
        dropout=0.1,
    )

    if pretrained_weights and Path(pretrained_weights).exists():
        state = torch.load(pretrained_weights, map_location="cpu")
        # Handle various checkpoint formats
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # Load with strict=False: encoder layers match, head layers may differ
        missing, unexpected = model.load_state_dict(state, strict=False)
        logging.info(f"Loaded pretrained weights: {Path(pretrained_weights).name}")
        if missing:
            logging.info(f"  Missing keys (new head): {len(missing)}")
        if unexpected:
            logging.info(f"  Unexpected keys (old head): {len(unexpected)}")
    elif pretrained_weights:
        logging.warning(
            f"Pretrained weights not found at {pretrained_weights}. "
            "Training from scratch."
        )

    return model


def set_encoder_frozen(model: nn.Module, frozen: bool) -> None:
    """Freeze or unfreeze the encoder (all layers except the final conv block)."""
    for name, param in model.named_parameters():
        is_output_block = "model.2" in name  # MONAI UNet: last decoder block
        param.requires_grad = (not frozen) or is_output_block
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    status = "frozen encoder" if frozen else "all layers unfrozen"
    logging.info(f"  [{status}] trainable params: {n_trainable:,}")


# ─── Trainer ──────────────────────────────────────────────────────────────────

class PanSegNetFinetuner:
    def __init__(self, cfg: dict, args: argparse.Namespace):
        self.cfg  = cfg
        self.args = args

        self.max_epochs     = int(cfg["approach_c"]["max_epochs"])
        self.freeze_epochs  = int(
            args.freeze_epochs if args.freeze_epochs is not None
            else cfg["approach_c"]["warmup_freeze_epochs"]
        )
        self.lr             = float(cfg["approach_c"]["lr"])
        self.patience       = int(cfg["approach_c"]["early_stop_patience"])
        self.output_dir     = Path(cfg["approach_c"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.patch_size     = cfg["approach_c"]["patch_size"]

        pretrained = cfg["approach_c"]["pretrained_weights"]
        self.model = build_model(cfg, pretrained).cuda()

        self.loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.max_epochs
        )

        self.sw_batch_size = args.sw_batch_size
        if _MONAI_AVAILABLE:
            self.train_loader, self.val_loader = build_monai_dataloaders(cfg, args)
        else:
            raise RuntimeError("MONAI is required for approach_c. Install: pip install monai")

        self.best_dice       = 0.0
        self.no_improve_cnt  = 0
        self.start_epoch     = 0

        if args.resume:
            self._load_checkpoint(args.resume)

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.start_epoch     = ckpt.get("epoch", 0)
        self.best_dice       = ckpt.get("best_dice", 0.0)
        self.no_improve_cnt  = ckpt.get("no_improve_cnt", 0)
        logging.info(f"Resumed from {path} (epoch {self.start_epoch}, best_dice={self.best_dice:.4f})")

    def _save_checkpoint(self, epoch: int, val_dice: float) -> None:
        path = self.output_dir / f"epoch_{epoch:03d}_dice_{val_dice:.4f}.pth"
        state = {
            "epoch":         epoch,
            "model":         self.model.state_dict(),
            "optimizer":     self.optimizer.state_dict(),
            "scheduler":     self.scheduler.state_dict(),
            "best_dice":     self.best_dice,
            "no_improve_cnt": self.no_improve_cnt,
        }
        torch.save(state, path)
        # Always keep a latest checkpoint for crash recovery
        torch.save(state, self.output_dir / "checkpoint_latest.pth")
        logging.info(f"  → Checkpoint: {path.name}")

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1} [train]",
                          leave=False):
            images = batch["image"].cuda().float()
            labels = batch["label"].cuda().long()
            preds  = self.model(images)
            loss   = self.loss_fn(preds, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / max(len(self.train_loader), 1)

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        dice_scores = []
        for batch in self.val_loader:
            images = batch["image"].cuda().float()
            labels = batch["label"].cpu().numpy()

            roi_size = self.patch_size
            preds = sliding_window_inference(
                images, roi_size=roi_size, sw_batch_size=self.sw_batch_size,
                predictor=self.model, overlap=0.25,
            )
            preds = torch.argmax(torch.softmax(preds, dim=1), dim=1)
            preds = preds.cpu().numpy()

            for pred, gt in zip(preds, labels.squeeze(1)):
                gt_bin = (gt > 0)
                if gt_bin.sum() > 0:
                    try:
                        m = calculate_metric_percase(pred > 0, gt_bin)
                        dice_scores.append(m[0])
                    except Exception:
                        pass

        return {"dice": float(np.mean(dice_scores)) if dice_scores else 0.0}

    def run(self) -> None:
        logging.info(f"Training for {self.max_epochs} epochs "
                     f"(encoder frozen for first {self.freeze_epochs})")

        for epoch in range(self.start_epoch, self.max_epochs):
            # Encoder freeze schedule
            if epoch < self.freeze_epochs:
                set_encoder_frozen(self.model, frozen=True)
            elif epoch == self.freeze_epochs:
                logging.info(f"Epoch {epoch+1}: Unfreezing encoder.")
                set_encoder_frozen(self.model, frozen=False)
                # Reset optimizer with all params
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=self.max_epochs - epoch
                )

            train_loss = self.train_epoch(epoch)
            val_metrics = self.validate()
            val_dice = val_metrics["dice"]

            self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]

            logging.info(
                f"Epoch {epoch+1:03d}/{self.max_epochs}: "
                f"loss={train_loss:.4f}  val_dice={val_dice:.4f}  lr={lr:.6f}"
            )

            if val_dice > self.best_dice:
                self.best_dice = val_dice
                self.no_improve_cnt = 0
                self._save_checkpoint(epoch + 1, val_dice)
                # Maintain a stable "best" symlink
                best_link = self.output_dir / "best_model.pth"
                if best_link.is_symlink():
                    best_link.unlink()
                best_link.symlink_to(
                    self.output_dir / f"epoch_{epoch+1:03d}_dice_{val_dice:.4f}.pth"
                )
            else:
                self.no_improve_cnt += 1
                # Still update latest checkpoint so crash recovery works
                torch.save({
                    "epoch":          epoch + 1,
                    "model":          self.model.state_dict(),
                    "optimizer":      self.optimizer.state_dict(),
                    "scheduler":      self.scheduler.state_dict(),
                    "best_dice":      self.best_dice,
                    "no_improve_cnt": self.no_improve_cnt,
                }, self.output_dir / "checkpoint_latest.pth")

            if self.no_improve_cnt >= self.patience:
                logging.info(
                    f"Early stopping at epoch {epoch+1} "
                    f"(no improvement for {self.patience} epochs). "
                    f"Best Dice: {self.best_dice:.4f}"
                )
                break

        logging.info(f"Training complete. Best validation Dice: {self.best_dice:.4f}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    output_dir = Path(cfg["approach_c"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "finetune.log"),
            logging.StreamHandler(),
        ],
    )

    trainer = PanSegNetFinetuner(cfg, args)
    trainer.run()


if __name__ == "__main__":
    main()
