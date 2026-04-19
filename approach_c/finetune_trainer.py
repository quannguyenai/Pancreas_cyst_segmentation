"""finetune_trainer.py — Fine-tune PanSegNet for pancreatic cyst segmentation.

No MONAI dependency.  Uses the faithful PanSegNet reimplementation in
approach_c/pansegnet.py and the existing nibabel-based Cyst dataset.

Architecture: exact Generic_TransUNet — nnUNet conv encoder/decoder +
SelfAtten3DBlock transformer bottleneck.  Pretrained PanSegNet weights
are loaded with key-exact matching; only the positional-encoding buffer
is re-initialised for our patch size (it is sinusoidal, not learned).

Training schedule
-----------------
1. Warmup (warmup_freeze_epochs): encoder frozen, decoder + transformer train.
2. Full fine-tune: all parameters unfreeze.

Usage
-----
python approach_c/finetune_trainer.py --config configs/paths.yaml
python approach_c/finetune_trainer.py --config configs/paths.yaml \\
    --resume approach_c/checkpoints/checkpoint_latest.pth
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.ndimage import zoom
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from configs import load_config
from comparison.utils.metrics import calculate_metric_percase
from pansegnet import PanSegNet, load_pansegnet_weights


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config",       default="configs/paths.yaml")
    p.add_argument("--resume",       default=None)
    p.add_argument("--freeze-epochs", type=int, default=None,
                   help="Override warmup_freeze_epochs from config.")
    p.add_argument("--gpu",          default="0")
    p.add_argument("--batch-size",   type=int, default=None)
    p.add_argument("--num-workers",  type=int, default=4)
    return p.parse_args()


# ─── Loss ─────────────────────────────────────────────────────────────────────

class DiceCELoss(nn.Module):
    """Dice + CrossEntropy on raw logits."""

    def __init__(self, n_classes: int = 2, smooth: float = 1e-5):
        super().__init__()
        self.n_classes = n_classes
        self.smooth    = smooth
        self.ce        = nn.CrossEntropyLoss()

    def _dice(self, probs: torch.Tensor, targets_oh: torch.Tensor) -> torch.Tensor:
        B, C = probs.shape[:2]
        p = probs.view(B, C, -1)
        t = targets_oh.view(B, C, -1).float()
        inter = (p * t).sum(-1)
        union = p.sum(-1) + t.sum(-1)
        dice  = (2 * inter + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : [B, C, D, H, W]  raw logits
        targets : [B, D, H, W]     integer class indices
        """
        probs    = torch.softmax(logits, dim=1)
        targets_oh = torch.zeros_like(probs)
        targets_oh.scatter_(1, targets.unsqueeze(1), 1)
        return self._dice(probs, targets_oh) + self.ce(logits, targets)


# ─── Sliding-window inference ─────────────────────────────────────────────────

@torch.no_grad()
def sliding_window_inference(
    model: nn.Module,
    volume: torch.Tensor,
    patch_size: list[int],
    overlap: float = 0.5,
    sw_batch_size: int = 2,
) -> torch.Tensor:
    """Tile a 3-D volume with overlapping patches; aggregate via Gaussian weights.

    Args
    ----
    model      : callable [B,1,pH,pW,pD] → [B,C,pH,pW,pD]
    volume     : [1, 1, H, W, D]  on any device
    patch_size : [pH, pW, pD]
    overlap    : fractional overlap between adjacent patches (0–1)
    """
    device = volume.device
    _, _, H, W, D = volume.shape
    pH, pW, pD = patch_size

    stride = [max(1, int(p * (1 - overlap))) for p in patch_size]

    # Pad so every axis is at least one patch and divisible by stride
    pad_h = max(0, pH - H)
    pad_w = max(0, pW - W)
    pad_d = max(0, pD - D)
    if pad_h or pad_w or pad_d:
        volume = torch.nn.functional.pad(
            volume, (0, pad_d, 0, pad_w, 0, pad_h), mode="constant", value=0
        )
    _, _, H2, W2, D2 = volume.shape

    # Run one dummy patch to get num_classes C
    with torch.no_grad():
        dummy = volume[:, :, :pH, :pW, :pD]
        was_training = model.training
        model.eval()
        C = model(dummy).shape[1]
        if was_training:
            model.train()

    accum  = torch.zeros(1, C, H2, W2, D2, device=device)
    weight = torch.zeros(1, 1, H2, W2, D2, device=device)

    # Gaussian importance window
    def _gaussian_1d(size: int) -> torch.Tensor:
        sigma = size / 6.0
        idx   = torch.arange(size, dtype=torch.float32)
        g     = torch.exp(-0.5 * ((idx - size / 2) / sigma) ** 2)
        return g / g.max()

    gw = (_gaussian_1d(pH)[:, None, None]
          * _gaussian_1d(pW)[None, :, None]
          * _gaussian_1d(pD)[None, None, :]).to(device)   # [pH, pW, pD]

    # Collect all patch positions
    def _indices(total, p, s):
        starts = list(range(0, total - p + 1, s))
        if not starts or starts[-1] + p < total:
            starts.append(total - p)
        return starts

    positions = [
        (i, j, k)
        for i in _indices(H2, pH, stride[0])
        for j in _indices(W2, pW, stride[1])
        for k in _indices(D2, pD, stride[2])
    ]

    # Process in mini-batches
    model.eval()
    for batch_start in range(0, len(positions), sw_batch_size):
        batch_pos   = positions[batch_start: batch_start + sw_batch_size]
        patches     = torch.stack([
            volume[0, :, i:i+pH, j:j+pW, k:k+pD] for i, j, k in batch_pos
        ])                                                  # [B, 1, pH, pW, pD]
        with torch.no_grad():
            outs = model(patches)                           # [B, C, pH, pW, pD]
            if isinstance(outs, tuple):
                outs = outs[-1]
        for idx, (i, j, k) in enumerate(batch_pos):
            accum [0, :, i:i+pH, j:j+pW, k:k+pD] += outs[idx] * gw
            weight[0, :, i:i+pH, j:j+pW, k:k+pD] += gw

    pred = accum / weight.clamp(min=1e-8)
    return pred[:, :, :H, :W, :D]   # un-pad


# ─── Data pipeline ────────────────────────────────────────────────────────────

def _normalize(img: np.ndarray) -> np.ndarray:
    """Per-volume percentile normalization for MRI (no fixed HU scale)."""
    p_low  = np.percentile(img, 0.5)
    p_high = np.percentile(img, 99.5)
    img = np.clip(img, p_low, p_high)
    denom = p_high - p_low
    if denom > 0:
        img = (img - p_low) / denom
    return img.astype(np.float32)


class CystPatchDataset(Dataset):
    """Load NIfTI CT volumes; return random foreground/background patches.

    During training, 50 % of crops are centred near a cyst voxel (positive
    mining) and 50 % are random.  Validation returns the full volume.
    """

    def __init__(
        self,
        txt_path: str | Path,
        patch_size: list[int],
        mode: str = "train",
        num_samples_per_volume: int = 4,
    ):
        self.patch_size  = patch_size
        self.mode        = mode
        self.samples_per = num_samples_per_volume

        with open(txt_path) as f:
            lines = f.readlines()[1:]
        self.pairs = [l.strip().split(",") for l in lines if l.strip()]
        print(f"[CystPatch/{mode}] {len(self.pairs)} volumes")

    def __len__(self) -> int:
        return len(self.pairs) * (self.samples_per if self.mode == "train" else 1)

    def __getitem__(self, idx: int) -> dict:
        vol_idx  = idx // self.samples_per if self.mode == "train" else idx
        img_path, mask_path = self.pairs[vol_idx]

        image = nib.load(img_path.strip()).get_fdata().astype(np.float32)
        label = nib.load(mask_path.strip()).get_fdata().astype(np.float32)
        image = _normalize(image)

        if self.mode == "train":
            image, label = self._random_crop(image, label)
            image, label = self._augment(image, label)
        # else: return full volume (for val sliding-window)

        img_t   = torch.from_numpy(image[None].astype(np.float32))  # [1,H,W,D]
        label_t = torch.from_numpy(label.astype(np.int64))          # [H,W,D]
        return {"image": img_t, "label": label_t, "path": img_path.strip()}

    # ── internal helpers ──────────────────────────────────────────────────────

    def _random_crop(self, img: np.ndarray, lbl: np.ndarray):
        pH, pW, pD = self.patch_size
        H, W, D    = img.shape

        # Pad if needed
        pad = [(max(0, pH - H), 0), (max(0, pW - W), 0), (max(0, pD - D), 0)]
        if any(p[0] for p in pad):
            img = np.pad(img, pad, mode="constant", constant_values=0)
            lbl = np.pad(lbl, pad, mode="constant", constant_values=0)
        H, W, D = img.shape

        if random.random() < 0.5 and lbl.sum() > 0:
            # Positive crop: centre near a random cyst voxel
            fg = np.argwhere(lbl > 0)
            ci, cj, ck = fg[random.randint(0, len(fg) - 1)]
            i0 = int(np.clip(ci - pH // 2, 0, H - pH))
            j0 = int(np.clip(cj - pW // 2, 0, W - pW))
            k0 = int(np.clip(ck - pD // 2, 0, D - pD))
        else:
            i0 = random.randint(0, H - pH)
            j0 = random.randint(0, W - pW)
            k0 = random.randint(0, D - pD)

        return img[i0:i0+pH, j0:j0+pW, k0:k0+pD], lbl[i0:i0+pH, j0:j0+pW, k0:k0+pD]

    def _augment(self, img: np.ndarray, lbl: np.ndarray):
        # Flips
        for axis in range(3):
            if random.random() < 0.5:
                img = np.flip(img, axis=axis).copy()
                lbl = np.flip(lbl, axis=axis).copy()
        # 90° rotations in each plane
        for axes in [(0, 1), (0, 2), (1, 2)]:
            k = random.randint(0, 3)
            if k:
                img = np.rot90(img, k=k, axes=axes).copy()
                lbl = np.rot90(lbl, k=k, axes=axes).copy()
        # Intensity jitter (CT HU-space: ±15 HU / scale ±10%)
        if random.random() < 0.5:
            img = img + random.uniform(-0.06, 0.06)  # ~±15HU after [−100,400] norm
        if random.random() < 0.5:
            img = img * random.uniform(0.9, 1.1)
        img = np.clip(img, 0.0, 1.0)
        return img, lbl


# ─── Trainer ──────────────────────────────────────────────────────────────────

class PanSegNetFinetuner:
    def __init__(self, cfg: dict, args: argparse.Namespace):
        self.cfg  = cfg
        self.args = args

        self.max_epochs    = int(cfg["approach_c"]["max_epochs"])
        self.freeze_epochs = int(
            args.freeze_epochs if args.freeze_epochs is not None
            else cfg["approach_c"]["warmup_freeze_epochs"]
        )
        self.lr         = float(cfg["approach_c"]["lr"])
        self.patience   = int(cfg["approach_c"]["early_stop_patience"])
        self.patch_size = cfg["approach_c"]["patch_size"]
        self.output_dir = Path(cfg["approach_c"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        batch_size  = args.batch_size or int(cfg["approach_c"]["batch_size"])
        num_workers = args.num_workers

        # ── Model ─────────────────────────────────────────────────────────────
        self.model = PanSegNet(
            input_channels=1,
            num_classes=2,
            patch_size=self.patch_size,
        ).cuda()

        pretrained = cfg["approach_c"]["pretrained_weights"]
        if pretrained and Path(pretrained).exists():
            load_pansegnet_weights(self.model, pretrained)
        else:
            logging.warning(f"Pretrained weights not found at {pretrained}. Training from scratch.")

        # ── Data ──────────────────────────────────────────────────────────────
        train_ds = CystPatchDataset(
            cfg["data"]["train_txt"], self.patch_size, mode="train",
            num_samples_per_volume=int(cfg["approach_c"].get("samples_per_volume", 4)),
        )
        val_ds = CystPatchDataset(
            cfg["data"]["val_txt"], self.patch_size, mode="val",
        )
        self.train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=num_workers,
        )

        # ── Optimiser & loss ──────────────────────────────────────────────────
        self.loss_fn   = DiceCELoss(n_classes=2)
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr, weight_decay=1e-4,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.max_epochs,
        )

        self.best_dice      = 0.0
        self.no_improve_cnt = 0
        self.start_epoch    = 0

        if args.resume:
            self._load_checkpoint(args.resume)

    # ── checkpoint I/O ────────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, val_dice: float) -> None:
        state = {
            "epoch":          epoch,
            "model":          self.model.state_dict(),
            "optimizer":      self.optimizer.state_dict(),
            "scheduler":      self.scheduler.state_dict(),
            "best_dice":      self.best_dice,
            "no_improve_cnt": self.no_improve_cnt,
        }
        path = self.output_dir / f"epoch_{epoch:03d}_dice_{val_dice:.4f}.pth"
        torch.save(state, path)
        torch.save(state, self.output_dir / "checkpoint_latest.pth")
        logging.info(f"  → saved {path.name}")

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.start_epoch    = ckpt.get("epoch", 0)
        self.best_dice      = ckpt.get("best_dice", 0.0)
        self.no_improve_cnt = ckpt.get("no_improve_cnt", 0)
        logging.info(f"Resumed from {path} (epoch {self.start_epoch}, best_dice={self.best_dice:.4f})")

    # ── training / validation ─────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total = 0.0
        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1} [train]", leave=False):
            images = batch["image"].cuda().float()   # [B,1,pH,pW,pD]
            labels = batch["label"].cuda().long()    # [B,pH,pW,pD]

            logits = self.model(images)
            if isinstance(logits, tuple):
                # logits[0] = finest (full patch), logits[1..] progressively coarser
                # weight 1.0 for finest, halved each step; downsample labels to match each output
                loss = torch.tensor(0.0, device=images.device)
                for i, lg in enumerate(logits):
                    w   = 1.0 / (2 ** i)
                    lbl = labels
                    if lg.shape[2:] != labels.shape[1:]:
                        lbl = torch.nn.functional.interpolate(
                            labels.float().unsqueeze(1), size=lg.shape[2:],
                            mode="nearest",
                        ).squeeze(1).long()
                    loss = loss + w * self.loss_fn(lg, lbl)
            else:
                loss = self.loss_fn(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total += loss.item()
        return total / max(len(self.train_loader), 1)

    @torch.no_grad()
    def _validate(self) -> float:
        self.model.eval()
        self.model.do_ds = False
        dice_scores: list[float] = []

        for batch in self.val_loader:
            volume = batch["image"].cuda().float()   # [1,1,H,W,D]
            gt     = batch["label"].squeeze(0).cpu().numpy()  # [H,W,D]

            pred_logits = sliding_window_inference(
                self.model, volume, self.patch_size, overlap=0.5, sw_batch_size=2,
            )
            pred = torch.argmax(pred_logits, dim=1).squeeze(0).cpu().numpy()

            if (gt > 0).sum() > 0:
                try:
                    m = calculate_metric_percase(pred > 0, gt > 0)
                    dice_scores.append(float(m[0]))
                except Exception:
                    pass

        self.model.do_ds = self.model._deep_supervision
        return float(np.mean(dice_scores)) if dice_scores else 0.0

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        logging.info(
            f"Training {self.max_epochs} epochs  "
            f"(encoder frozen for first {self.freeze_epochs})"
        )

        for epoch in range(self.start_epoch, self.max_epochs):
            if epoch < self.freeze_epochs:
                self.model.set_encoder_frozen(frozen=True)
            elif epoch == self.freeze_epochs:
                logging.info(f"Epoch {epoch+1}: unfreezing encoder.")
                self.model.set_encoder_frozen(frozen=False)
                self.optimizer = optim.AdamW(
                    self.model.parameters(), lr=self.lr, weight_decay=1e-4,
                )
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=self.max_epochs - epoch,
                )

            train_loss = self._train_epoch(epoch)
            val_dice   = self._validate()
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]

            logging.info(
                f"Epoch {epoch+1:03d}/{self.max_epochs}  "
                f"loss={train_loss:.4f}  val_dice={val_dice:.4f}  lr={lr:.2e}"
            )

            if val_dice > self.best_dice:
                self.best_dice      = val_dice
                self.no_improve_cnt = 0
                self._save_checkpoint(epoch + 1, val_dice)
                best_link = self.output_dir / "best_model.pth"
                if best_link.is_symlink():
                    best_link.unlink()
                best_link.symlink_to(
                    self.output_dir / f"epoch_{epoch+1:03d}_dice_{val_dice:.4f}.pth"
                )
            else:
                self.no_improve_cnt += 1
                state = {
                    "epoch":          epoch + 1,
                    "model":          self.model.state_dict(),
                    "optimizer":      self.optimizer.state_dict(),
                    "scheduler":      self.scheduler.state_dict(),
                    "best_dice":      self.best_dice,
                    "no_improve_cnt": self.no_improve_cnt,
                }
                torch.save(state, self.output_dir / "checkpoint_latest.pth")

            if self.no_improve_cnt >= self.patience:
                logging.info(
                    f"Early stopping (no improvement for {self.patience} epochs). "
                    f"Best Dice: {self.best_dice:.4f}"
                )
                break

        logging.info(f"Training complete. Best val Dice: {self.best_dice:.4f}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)
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
