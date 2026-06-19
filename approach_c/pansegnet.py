"""pansegnet.py — Faithful 3-D reimplementation of PanSegNet (Generic_TransUNet).

Ported from https://github.com/NUBagciLab/PaNSegNet (Apache 2.0).

Every module *attribute name* is kept identical to the original so that
``model.load_state_dict(checkpoint, strict=False)`` can load pretrained weights
with only the seg-head keys missing (when num_classes differs) or with the
pos_encode.pe buffer silently replaced for a different patch size.

Architecture overview
---------------------
Encoder  : StackedConvLayers × (num_pool stages), downsampled by stride in the
           first conv of each stage (convolutional_pooling=True).
Bottleneck: StackedConvLayers → SelfAtten3DBlock (linear self-attention).
Decoder  : ConvTranspose3d upsample + skip-concat + StackedConvLayers,
           one seg_output head per decoder stage (deep-supervision ready).

Default hyper-parameters match nnUNet 3-D fullres for pancreas:
  base_num_features=32, num_pool=5, pool_kernels=2×2×2, conv_per_stage=2.
"""

from __future__ import annotations

import copy
import math
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ─── Positional encoding ──────────────────────────────────────────────────────

class AbsolutePosEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, max_length: int, embedding_dim: int):
        super().__init__()
        self.max_length   = max_length
        self.embedding_dim = embedding_dim
        pe       = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-math.log(10000.0) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))   # [1, max_length, d]

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe


# ─── Transformer blocks ───────────────────────────────────────────────────────

def _clones(module: nn.Module, N: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _linear_attention(
    query: Tensor, key: Tensor, value: Tensor,
    mask: Optional[Tensor] = None,
    dropout: Optional[nn.Module] = None,
) -> Tensor:
    d_model = query.size(-1)
    query   = F.softmax(query, dim=-1) / math.sqrt(d_model)
    if mask is not None:
        key   = key.masked_fill_(~mask, -1e9)
        value = value.masked_fill_(~mask, 0)
    key     = F.softmax(key, dim=-2)
    context = torch.einsum("bhnd,bhne->bhde", key, value)
    if dropout is not None:
        query = dropout(query)
    return torch.einsum("bhnd,bhde->bhne", query, context)


class MultihAttention(nn.Module):
    """Multi-head linear self-attention.
    Class name kept as 'MultihAttention' to match PanSegNet state-dict keys.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.d_k     = d_model // nhead
        self.nhead   = nhead
        self.linears = _clones(nn.Linear(d_model, d_model), 4)
        self.attn    = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: Tensor, key: Tensor, value: Tensor,
        src_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if src_mask is not None:
            src_mask = src_mask.unsqueeze(1)
        n_batch = query.size(0)
        q, k, v = [
            lin(x).view(n_batch, -1, self.nhead, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]
        x = _linear_attention(q, k, v, mask=src_mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.nhead * self.d_k)
        return self.linears[-1](x)


class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model: int, nhead: int, dim_feedforward: int,
        dropout: float = 0.1, activation: str = "gelu",
        layer_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.d_model   = d_model
        self.self_attn = MultihAttention(d_model, nhead, dropout=dropout)
        self.linear1   = nn.Linear(d_model, dim_feedforward)
        self.dropout   = nn.Dropout(p=dropout)
        self.linear2   = nn.Linear(dim_feedforward, d_model)
        self.layer_norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1  = nn.Dropout(p=dropout)
        self.dropout2  = nn.Dropout(p=dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, x: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        x = self.layer_norm1(x + self.dropout1(self.self_attn(x, x, x, src_mask)))
        x = self.layer_norm2(x + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x))))))
        return x


class TransEncoder(nn.Module):
    def __init__(self, attn_layer: nn.Module, N: int):
        super().__init__()
        self.layers = _clones(attn_layer, N)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x


class SelfAtten3DBlock(nn.Module):
    """Transformer bottleneck for 3-D feature maps.

    Attribute names match PanSegNet for weight loading:
      linear_proj, linear_back_proj, pos_encode, transformer.
    """

    def __init__(
        self,
        in_dim: int,
        feature_length: int,
        d_model: int,
        nhead: int,
        dropout: float = 0.3,
        N: int = 8,
    ):
        super().__init__()
        self.in_dim         = in_dim
        self.d_model        = d_model
        self.feature_length = feature_length
        self.linear_proj      = nn.Linear(in_dim,   d_model)
        self.linear_back_proj = nn.Linear(d_model,  in_dim)
        self.pos_encode       = AbsolutePosEncoding(feature_length, d_model)
        attn_layer            = SelfAttentionLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model, dropout=dropout,
        )
        self.transformer = TransEncoder(attn_layer, N)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        B, _, H, W, D = x.shape
        x = x.flatten(2).transpose(1, 2)        # [B, H*W*D, in_dim]
        x = self.linear_proj(x)                  # [B, H*W*D, d_model]
        x = self.pos_encode(x)
        if mask is not None:
            mask = mask.flatten(2).transpose(1, 2)
        x = self.transformer(x, mask=mask)
        x = self.linear_back_proj(x)             # [B, H*W*D, in_dim]
        x = x.transpose(1, 2).reshape(B, -1, H, W, D)
        return x


# ─── Conv building blocks ─────────────────────────────────────────────────────

class ConvDropoutNormNonlin(nn.Module):
    """Conv3d → (Dropout) → InstanceNorm3d → LeakyReLU.
    Attribute names: conv, dropout, instnorm, lrelu — match PanSegNet.
    """

    def __init__(
        self,
        input_channels: int, output_channels: int,
        conv_op=nn.Conv3d, conv_kwargs: Optional[dict] = None,
        norm_op=nn.InstanceNorm3d, norm_op_kwargs: Optional[dict] = None,
        dropout_op=nn.Dropout3d, dropout_op_kwargs: Optional[dict] = None,
        nonlin=nn.LeakyReLU, nonlin_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        if nonlin_kwargs    is None: nonlin_kwargs    = {"negative_slope": 1e-2, "inplace": True}
        if dropout_op_kwargs is None: dropout_op_kwargs = {"p": 0.5, "inplace": True}
        if norm_op_kwargs   is None: norm_op_kwargs   = {"eps": 1e-5, "affine": True}
        if conv_kwargs      is None: conv_kwargs      = {"kernel_size": 3, "stride": 1, "padding": 1,
                                                          "dilation": 1, "bias": True}
        self.conv     = conv_op(input_channels, output_channels, **conv_kwargs)
        p = dropout_op_kwargs.get("p", 0)
        self.dropout  = dropout_op(**dropout_op_kwargs) if (dropout_op is not None and p is not None and p > 0) else None
        self.instnorm = norm_op(output_channels, **norm_op_kwargs)
        self.lrelu    = nonlin(**nonlin_kwargs)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))


class StackedConvLayers(nn.Module):
    """N stacked ConvDropoutNormNonlin blocks; first can use a stride for downsampling.
    Attribute names: blocks, input_channels, output_channels — match PanSegNet.
    """

    def __init__(
        self,
        input_feature_channels: int, output_feature_channels: int, num_convs: int,
        conv_op=nn.Conv3d, conv_kwargs: Optional[dict] = None,
        norm_op=nn.InstanceNorm3d, norm_op_kwargs: Optional[dict] = None,
        dropout_op=nn.Dropout3d, dropout_op_kwargs: Optional[dict] = None,
        nonlin=nn.LeakyReLU, nonlin_kwargs: Optional[dict] = None,
        first_stride=None,
    ):
        self.input_channels  = input_feature_channels
        self.output_channels = output_feature_channels

        if nonlin_kwargs    is None: nonlin_kwargs    = {"negative_slope": 1e-2, "inplace": True}
        if dropout_op_kwargs is None: dropout_op_kwargs = {"p": 0.5, "inplace": True}
        if norm_op_kwargs   is None: norm_op_kwargs   = {"eps": 1e-5, "affine": True}
        if conv_kwargs      is None: conv_kwargs      = {"kernel_size": 3, "stride": 1, "padding": 1,
                                                          "dilation": 1, "bias": True}
        super().__init__()

        first_conv_kwargs = dict(conv_kwargs)
        if first_stride is not None:
            first_conv_kwargs["stride"] = first_stride

        self.blocks = nn.Sequential(
            ConvDropoutNormNonlin(
                input_feature_channels, output_feature_channels,
                conv_op, first_conv_kwargs,
                norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
            ),
            *[
                ConvDropoutNormNonlin(
                    output_feature_channels, output_feature_channels,
                    conv_op, conv_kwargs,
                    norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                )
                for _ in range(num_convs - 1)
            ],
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


# ─── PanSegNet ────────────────────────────────────────────────────────────────

_MAX_NUM_FILTERS = 320   # nnUNet default cap for 3-D


class PanSegNet(nn.Module):
    """3-D PanSegNet — nnUNet encoder/decoder + SelfAtten3DBlock bottleneck.

    Default parameters match nnUNet 3-D fullres for pancreas:
      base_num_features=32, num_pool=5, feat_mul=2, conv_per_stage=2,
      convolutional_pooling=True, convolutional_upsampling=True.

    Module attribute names are identical to the original Generic_TransUNet so
    pretrained PanSegNet.pth weights load correctly via load_state_dict.

    Args
    ----
    input_channels      : CT = 1
    base_num_features   : starting feature count (32)
    num_classes         : output segmentation classes (2 for binary)
    num_pool            : number of downsampling stages (5)
    patch_size          : input patch dimensions; used to compute attention sequence length
    pool_op_kernel_sizes: per-stage pool/stride kernel, e.g. [[2,2,2]]*5
    conv_kernel_sizes   : per-stage conv kernel, e.g. [[3,3,3]]*6
    conv_per_stage      : convolutions per encoder/decoder stage (2)
    deep_supervision    : return auxiliary seg outputs for training (True)
    """

    def __init__(
        self,
        input_channels: int = 1,
        base_num_features: int = 32,
        num_classes: int = 2,
        num_pool: int = 5,
        patch_size: Sequence[int] = (128, 128, 128),
        pool_op_kernel_sizes: Optional[list] = None,
        conv_kernel_sizes: Optional[list] = None,
        conv_per_stage: int = 2,
        feat_map_mul_on_downscale: int = 2,
        max_num_features: int = _MAX_NUM_FILTERS,
        dropout_p: float = 0.0,
        deep_supervision: bool = True,
        # transformer params
        transformer_d_model: int = 512,
        transformer_nhead: int = 8,
        transformer_N: int = 8,
        transformer_dropout: float = 0.3,
    ):
        super().__init__()

        if pool_op_kernel_sizes is None:
            pool_op_kernel_sizes = [[2, 2, 2]] * num_pool
        if conv_kernel_sizes is None:
            conv_kernel_sizes = [[3, 3, 3]] * (num_pool + 1)

        self._deep_supervision = deep_supervision
        self.do_ds             = deep_supervision   # toggled False at inference
        self.patch_size        = list(patch_size)
        self.num_pool          = num_pool

        # shared conv/norm kwargs
        norm_kwargs    = {"eps": 1e-5, "affine": True}
        dropout_kwargs = {"p": dropout_p}

        def _conv_kwargs(kernel):
            k = list(kernel)
            p = [ki // 2 for ki in k]
            return {"kernel_size": k, "stride": 1, "padding": p, "dilation": 1, "bias": True}

        # ── Encoder ──────────────────────────────────────────────────────────
        self.conv_blocks_context = []
        self.td                  = []   # empty; pooling via stride (convolutional_pooling=True)

        out_ch = base_num_features
        in_ch  = input_channels

        for d in range(num_pool):
            first_stride = pool_op_kernel_sizes[d - 1] if d != 0 else None
            ck = _conv_kwargs(conv_kernel_sizes[d])
            self.conv_blocks_context.append(
                StackedConvLayers(
                    in_ch, out_ch, conv_per_stage,
                    nn.Conv3d, ck,
                    nn.InstanceNorm3d, dict(norm_kwargs),
                    nn.Dropout3d, dict(dropout_kwargs),
                    nn.LeakyReLU, {"negative_slope": 1e-2, "inplace": True},
                    first_stride=first_stride,
                )
            )
            in_ch  = out_ch
            out_ch = min(int(round(out_ch * feat_map_mul_on_downscale)), max_num_features)

        # ── Bottleneck ───────────────────────────────────────────────────────
        # convolutional_upsampling=True → final_num_features = out_ch (not capped yet)
        final_ch = out_ch   # = min(320*2, 320) = 320 for default config
        ck_bn    = _conv_kwargs(conv_kernel_sizes[num_pool])
        self.conv_blocks_context.append(nn.Sequential(
            StackedConvLayers(
                in_ch, out_ch, conv_per_stage - 1,
                nn.Conv3d, ck_bn,
                nn.InstanceNorm3d, dict(norm_kwargs),
                nn.Dropout3d, dict(dropout_kwargs),
                nn.LeakyReLU, {"negative_slope": 1e-2, "inplace": True},
                first_stride=pool_op_kernel_sizes[-1],
            ),
            StackedConvLayers(
                out_ch, final_ch, 1,
                nn.Conv3d, ck_bn,
                nn.InstanceNorm3d, dict(norm_kwargs),
                nn.Dropout3d, dict(dropout_kwargs),
                nn.LeakyReLU, {"negative_slope": 1e-2, "inplace": True},
            ),
        ))

        self.last_dim_features = final_ch

        # ── Transformer bottleneck ───────────────────────────────────────────
        cum_up       = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
        features_num = int(np.prod(patch_size) / np.prod(cum_up[0]))
        self.self_atten = SelfAtten3DBlock(
            in_dim=self.last_dim_features,
            d_model=transformer_d_model,
            feature_length=features_num,
            nhead=transformer_nhead,
            dropout=transformer_dropout,
            N=transformer_N,
        )

        # ── Decoder ──────────────────────────────────────────────────────────
        self.conv_blocks_localization = []
        self.tu                       = []
        self.seg_outputs              = []

        nfeatures_from_down = final_ch

        for u in range(num_pool):
            skip_ch = self.conv_blocks_context[-(2 + u)].output_channels
            concat_ch = skip_ch * 2
            # convolutional_upsampling=True → final_num_features = skip_ch for all u
            final_num_features = skip_ch

            pool_size = pool_op_kernel_sizes[-(u + 1)]
            self.tu.append(
                nn.ConvTranspose3d(
                    nfeatures_from_down, skip_ch,
                    kernel_size=pool_size, stride=pool_size, bias=False,
                )
            )

            ck_dec = _conv_kwargs(conv_kernel_sizes[-(u + 1)])
            self.conv_blocks_localization.append(nn.Sequential(
                StackedConvLayers(
                    concat_ch, skip_ch, conv_per_stage - 1,
                    nn.Conv3d, ck_dec,
                    nn.InstanceNorm3d, dict(norm_kwargs),
                    nn.Dropout3d, dict(dropout_kwargs),
                    nn.LeakyReLU, {"negative_slope": 1e-2, "inplace": True},
                ),
                StackedConvLayers(
                    skip_ch, final_num_features, 1,
                    nn.Conv3d, ck_dec,
                    nn.InstanceNorm3d, dict(norm_kwargs),
                    nn.Dropout3d, dict(dropout_kwargs),
                    nn.LeakyReLU, {"negative_slope": 1e-2, "inplace": True},
                ),
            ))
            self.seg_outputs.append(
                nn.Conv3d(final_num_features, num_classes, kernel_size=1, stride=1,
                          padding=0, dilation=1, groups=1, bias=False)
            )
            nfeatures_from_down = final_num_features

        # deep-supervision upscaling ops (identity since upscale_logits=False)
        self.upscale_logits_ops = [lambda x: x] * (num_pool - 1)

        # ── Register as ModuleLists ───────────────────────────────────────────
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.conv_blocks_context      = nn.ModuleList(self.conv_blocks_context)
        self.td                       = nn.ModuleList(self.td)   # empty
        self.tu                       = nn.ModuleList(self.tu)
        self.seg_outputs              = nn.ModuleList(self.seg_outputs)

        # weight init
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, a=1e-2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, a=1e-2)

    # ── helpers ───────────────────────────────────────────────────────────────

    @property
    def _encoder_module_names(self) -> set[str]:
        """Parameter name prefixes belonging to the encoder conv blocks."""
        return {"conv_blocks_context", "td"}

    def set_encoder_frozen(self, frozen: bool) -> None:
        """Freeze/unfreeze encoder; transformer + decoder always stay trainable."""
        for name, param in self.named_parameters():
            is_encoder = any(name.startswith(p) for p in self._encoder_module_names)
            param.requires_grad = (not frozen) or (not is_encoder)
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        import logging
        logging.info(f"  [{'frozen encoder' if frozen else 'all unfrozen'}] trainable params: {n:,}")

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: Tensor) -> Tensor | tuple:
        skips       = []
        seg_outputs = []

        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            # td is empty (convolutional_pooling); no separate pool op

        x = self.conv_blocks_context[-1](x)   # bottleneck conv
        x = self.self_atten(x)                # transformer

        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            seg_outputs.append(self.seg_outputs[u](x))   # raw logits; softmax applied by loss/inference

        if self._deep_supervision and self.do_ds:
            return tuple(
                [seg_outputs[-1]]
                + [op(s) for op, s in
                   zip(self.upscale_logits_ops[:-1], seg_outputs[:-1][::-1])]
            )
        return seg_outputs[-1]


# ─── Weight loading helper ────────────────────────────────────────────────────

def load_pansegnet_weights(
    model: PanSegNet,
    checkpoint_path: str,
    num_classes_ours: int = 2,
) -> None:
    """Load PanSegNet pretrained weights into model.

    Strategy
    --------
    1. Load checkpoint; handle {'model': ...} / {'state_dict': ...} wrappers.
    2. Pop self_atten.pos_encode.pe — it is sinusoidal (not learned) and its
       size depends on the pretrain patch_size which may differ from ours.
       Our model already has the correct pe for its patch_size.
    3. Pop seg_outputs keys if num_classes differs (shape mismatch).
    4. load_state_dict strict=False — encoder + transformer load fully;
       any remaining shape mismatches are skipped with a warning.
    """
    import logging
    from pathlib import Path

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        # nnUNet v1 .model files use 'state_dict'; .pth files may use 'model'
        state = ckpt.get("state_dict", ckpt.get("model", ckpt))
    else:
        state = ckpt

    # Remove positional encoding buffer (sinusoidal — size depends on pretrain patch_size)
    state.pop("self_atten.pos_encode.pe", None)

    # Remove ALL shape-mismatched keys (e.g. anisotropic [1,3,3] kernels from MRI model)
    our_state = model.state_dict()
    skipped = []
    for k in list(state.keys()):
        if k in our_state and state[k].shape != our_state[k].shape:
            skipped.append(f"{k}: ckpt{tuple(state[k].shape)} vs ours{tuple(our_state[k].shape)}")
            state.pop(k)
    if skipped:
        logging.info(f"  Skipped {len(skipped)} shape-mismatched keys (anisotropic→isotropic): "
                     + ", ".join(skipped[:3]) + ("..." if len(skipped) > 3 else ""))

    missing, unexpected = model.load_state_dict(state, strict=False)
    logging.info(
        f"Loaded PanSegNet weights from {Path(checkpoint_path).name}\n"
        f"  missing  : {len(missing)}   (new layers / shape mismatches)\n"
        f"  unexpected: {len(unexpected)}  (old layers not in our model)"
    )
    if missing:
        logging.debug("  Missing keys: " + ", ".join(missing[:10]))
