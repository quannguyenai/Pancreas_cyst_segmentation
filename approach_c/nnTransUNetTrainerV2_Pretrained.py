"""nnTransUNetTrainerV2_Pretrained — PanSegNet fine-tuning trainer for nnUNet v1.

Inherits everything from nnTransUNetTrainerV2 (full nnUNet augmentation pipeline,
preprocessing, LR schedule, checkpointing) and adds:

  1. Pretrained weight loading from pansegnet_weights/averaged_T1T2.model
     (or any path set via PANSEGNET_PRETRAINED_WEIGHTS env var).
  2. Encoder freeze warmup: encoder is frozen for the first
     PANSEGNET_WARMUP_EPOCHS epochs, then unfrozen for full fine-tuning.

Usage
-----
  export PANSEGNET_PRETRAINED_WEIGHTS=/path/to/averaged_T1T2.model
  export PANSEGNET_WARMUP_EPOCHS=25
  nnUNet_train 3d_fullres nnTransUNetTrainerV2_Pretrained 1 0
"""

import os
import torch
from nnunet.training.network_training.nnTransUNetTrainerV2 import nnTransUNetTrainerV2

WARMUP_FREEZE_EPOCHS = int(os.environ.get("PANSEGNET_WARMUP_EPOCHS", "25"))
PRETRAINED_WEIGHTS   = os.environ.get("PANSEGNET_PRETRAINED_WEIGHTS", "")


class nnTransUNetTrainerV2_Pretrained(nnTransUNetTrainerV2):

    def initialize(self, training=True, force_load_plans=False):
        super().initialize(training, force_load_plans)
        if not training:
            return
        # When resuming, the checkpoint is loaded AFTER initialize() by run_training().
        # Skip pretrained init to avoid re-freezing the encoder mid-training.
        ckpt_exists = (
            os.path.isfile(os.path.join(self.output_folder, "model_latest.model")) or
            os.path.isfile(os.path.join(self.output_folder, "model_final_checkpoint.model"))
        )
        if ckpt_exists:
            self.print_to_log_file("Checkpoint found — skipping pretrained weight loading.")
            return
        if PRETRAINED_WEIGHTS and os.path.isfile(PRETRAINED_WEIGHTS):
            self._load_pretrained_weights(PRETRAINED_WEIGHTS)
            self._set_encoder_frozen(True)
            self.print_to_log_file(
                f"Pretrained weights: {PRETRAINED_WEIGHTS}\n"
                f"Encoder frozen for first {WARMUP_FREEZE_EPOCHS} epochs."
            )
        else:
            self.print_to_log_file(
                "WARNING: PANSEGNET_PRETRAINED_WEIGHTS not set or not found. "
                "Training from scratch."
            )

    def maybe_update_lr(self, epoch=None):
        super().maybe_update_lr(epoch)
        if self.epoch == WARMUP_FREEZE_EPOCHS:
            self._set_encoder_frozen(False)
            self.print_to_log_file(
                f"Epoch {self.epoch}: encoder unfrozen — full fine-tune begins."
            )

    def _load_pretrained_weights(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            state = ckpt.get("state_dict", ckpt.get("model", ckpt))
        else:
            state = ckpt

        # Positional encoding size depends on pretrain patch size — let model reinit it
        state.pop("self_atten.pos_encode.pe", None)

        # Skip shape-mismatched keys (e.g. anisotropic [1,3,3] kernels from MRI model)
        our_state = self.network.state_dict()
        skipped = [k for k in list(state)
                   if k in our_state and state[k].shape != our_state[k].shape]
        for k in skipped:
            state.pop(k)

        missing, unexpected = self.network.load_state_dict(state, strict=False)
        self.print_to_log_file(
            f"Weights loaded — missing: {len(missing)}, "
            f"unexpected: {len(unexpected)}, "
            f"shape-skipped: {len(skipped)}"
        )

    def _set_encoder_frozen(self, frozen: bool) -> None:
        for name, param in self.network.named_parameters():
            is_encoder = (name.startswith("conv_blocks_context")
                          or name.startswith("td"))
            param.requires_grad = (not frozen) if is_encoder else True
        n = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        self.print_to_log_file(
            f"{'Encoder frozen' if frozen else 'All params unfrozen'} "
            f"— trainable params: {n:,}"
        )
