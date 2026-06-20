"""3D TransUNet baseline.

Vendored from the official repo Beckschen/3D-TransUNet
(nn_transunet/networks/, HEAD 9f18182), trimmed to the classic-TransUNet
configuration: a 3D nnU-Net-style CNN encoder/decoder with a ViT transformer at
the bottleneck (``is_max=False``, ``is_max_bottleneck_transformer=True``). The
query-based Mask2Former decoder (``is_max=True``) and its ``mask2former_modeling``
subtree are NOT vendored — they are only imported lazily inside the unused
``is_max`` branch.

``build_transunet3d`` returns a model whose forward emits raw logits (the
``final_nonlin`` softmax is disabled so the harness's CE + softmax-Dice loss sees
logits, not probabilities). Because the segmentation heads only exist when
``deep_supervision=True``, we keep DS heads enabled and select the full-resolution
head (index 0 of the returned tuple) — handled by the ``net_factory`` wrapper.
"""

from __future__ import annotations

import torch.nn as nn

from .transunet3d_model import Generic_TransUNet_max_ppbp


def _identity(x):
    return x


def build_transunet3d(
    in_channels: int = 1,
    num_classes: int = 2,
    patch_size=(96, 96, 96),
    base_num_features: int = 32,
    num_pool: int = 5,
    vit_depth: int = 12,
    vit_hidden_size: int = 768,
    vit_mlp_dim: int = 3072,
    vit_num_heads: int = 12,
) -> nn.Module:
    """Build a 3D TransUNet (CNN U-Net + ViT bottleneck).

    forward() returns a tuple of deep-supervision logits; element 0 is the
    full-resolution head. The net_factory wraps this and picks element 0.

    At patch_size=(96,96,96) with num_pool=5 isotropic stride-2 pools the
    bottleneck grid is 3x3x3 (27 tokens) which feeds the ViT transformer.
    """
    return Generic_TransUNet_max_ppbp(
        input_channels=in_channels,
        base_num_features=base_num_features,
        num_classes=num_classes,
        num_pool=num_pool,
        num_conv_per_stage=2,
        conv_op=nn.Conv3d,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=nn.Dropout3d,
        dropout_op_kwargs={"p": 0.0, "inplace": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"negative_slope": 1e-2, "inplace": True},
        deep_supervision=True,          # required: seg heads only built when do_ds
        final_nonlin=_identity,         # raw logits for CE + softmax-Dice loss
        pool_op_kernel_sizes=[[2, 2, 2]] * num_pool,
        conv_kernel_sizes=[[3, 3, 3]] * (num_pool + 1),
        convolutional_pooling=True,
        convolutional_upsampling=True,
        patch_size=list(patch_size),
        is_vit_pretrain=False,          # train from scratch, no external pretraining
        vit_depth=vit_depth,
        vit_hidden_size=vit_hidden_size,
        vit_mlp_dim=vit_mlp_dim,
        vit_num_heads=vit_num_heads,
        is_max_bottleneck_transformer=True,
        is_max=False,                   # disable Mask2Former query decoder
        is_max_ms=False,
    )
