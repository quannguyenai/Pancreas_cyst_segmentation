"""net_factory.py — Unified model factory for all comparison baselines."""

from __future__ import annotations

import torch.nn as nn


def net_factory(
    net_type: str,
    in_chns: int = 1,
    class_num: int = 2,
    **kwargs,
) -> nn.Module:
    """Instantiate a segmentation network by name.

    Parameters
    ----------
    net_type : str
        One of: ``"unet_2d"``, ``"unet_3d"``, ``"vnet"``, ``"unetr"``.
    in_chns : int
        Number of input channels (default 1 for CT).
    class_num : int
        Number of output classes (default 2 for binary segmentation).
    **kwargs
        Extra keyword arguments forwarded to the network constructor.

    Returns
    -------
    nn.Module
    """
    net_type = net_type.lower()

    if net_type == "unet_2d":
        from comparison.networks.unet import UNet
        return UNet(
            n_channels=in_chns,
            n_classes=class_num,
            bilinear=kwargs.get("bilinear", True),
        )

    if net_type in ("unet_3d", "unet3d"):
        from comparison.networks.Unet3D import UNet3D
        return UNet3D(
            in_channels=in_chns,
            out_channels=class_num,
            **{k: v for k, v in kwargs.items() if k != "bilinear"},
        )

    if net_type == "vnet":
        from comparison.networks.VNet import VNet
        return VNet(
            n_channels=in_chns,
            n_classes=class_num,
            normalization=kwargs.get("normalization", "batchnorm"),
            has_dropout=kwargs.get("has_dropout", True),
        )

    if net_type == "unetr":
        from comparison.networks.unetr import UNETR
        return UNETR(
            in_channels=in_chns,
            out_channels=class_num,
            img_size=kwargs.get("img_size", (96, 96, 96)),
            feature_size=kwargs.get("feature_size", 16),
            hidden_size=kwargs.get("hidden_size", 768),
            mlp_dim=kwargs.get("mlp_dim", 3072),
            num_heads=kwargs.get("num_heads", 12),
            pos_embed=kwargs.get("pos_embed", "perceptron"),
            norm_name=kwargs.get("norm_name", "instance"),
            conv_block=kwargs.get("conv_block", True),
            res_block=kwargs.get("res_block", True),
            dropout_rate=kwargs.get("dropout_rate", 0.0),
        )

    raise ValueError(
        f"Unknown net_type={net_type!r}. "
        "Choose from: unet_2d, unet_3d, vnet, unetr."
    )
