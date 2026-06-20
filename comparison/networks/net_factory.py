"""net_factory.py — Unified model factory for all comparison baselines.

Every model is wrapped in ``TupleOut`` so ``forward`` returns a uniform
``(logits, logits)`` 2-tuple — satisfying both the train loop (which unwraps the
tuple) and the sliding-window test path (which does ``y1, _ = model(x)``). See
``comparison/networks/_wrap.py``.

Architectures
-------------
Vendored (pure-torch, in ``comparison/networks/``):
    unet_2d, unet_3d/unet3d, vnet, unetr, transunet (3D, from Beckschen/3D-TransUNet)
MONAI built-ins (``monai.networks.nets``):
    unetpp (BasicUNetPlusPlus), swinunetr (SwinUNETR), mednext (MedNeXt)
"""

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
        One of: ``unet_2d``, ``unet_3d``, ``vnet``, ``unetr``,
        ``unetpp``, ``transunet``, ``swinunetr``, ``mednext``.
    in_chns : int
        Number of input channels (default 1 for single-modality MRI).
    class_num : int
        Number of output classes (default 2 for binary segmentation).
    **kwargs
        Extra keyword arguments forwarded to the network constructor.

    Returns
    -------
    nn.Module
        Wrapped so ``forward(x)`` returns ``(logits, logits)``.
    """
    from comparison.networks._wrap import TupleOut

    net_type = net_type.lower()

    if net_type == "unet_2d":
        from comparison.networks.unet import UNet
        return TupleOut(UNet(
            n_channels=in_chns,
            n_classes=class_num,
            bilinear=kwargs.get("bilinear", True),
        ))

    if net_type in ("unet_3d", "unet3d"):
        # Unet3D.py defines the class as `UNet`, not `UNet3D`.
        from comparison.networks.Unet3D import UNet as UNet3D
        return TupleOut(UNet3D(
            in_dim=in_chns,
            out_dim=class_num,
            **{k: v for k, v in kwargs.items() if k != "bilinear"},
        ))

    if net_type == "vnet":
        from comparison.networks.VNet import VNet
        return TupleOut(VNet(
            n_channels=in_chns,
            n_classes=class_num,
            normalization=kwargs.get("normalization", "batchnorm"),
            has_dropout=kwargs.get("has_dropout", True),
        ))

    if net_type == "unetr":
        from comparison.networks.unetr import UNETR
        return TupleOut(UNETR(
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
        ))

    if net_type in ("unetpp", "unet++", "unetplusplus"):
        from monai.networks.nets import BasicUNetPlusPlus
        net = BasicUNetPlusPlus(
            spatial_dims=kwargs.get("spatial_dims", 3),
            in_channels=in_chns,
            out_channels=class_num,
            features=kwargs.get("features", (32, 32, 64, 128, 256, 32)),
            deep_supervision=False,
        )
        # BasicUNetPlusPlus always returns a list (final full-res head at [0]).
        return TupleOut(net, pick=lambda o: o[0])

    if net_type == "swinunetr":
        from monai.networks.nets import SwinUNETR
        net = SwinUNETR(
            in_channels=in_chns,
            out_channels=class_num,
            spatial_dims=kwargs.get("spatial_dims", 3),
            feature_size=kwargs.get("feature_size", 48),  # must be divisible by 12
            use_checkpoint=kwargs.get("use_checkpoint", False),
        )
        return TupleOut(net)

    if net_type == "mednext":
        from monai.networks.nets import MedNeXt
        net = MedNeXt(
            spatial_dims=kwargs.get("spatial_dims", 3),
            in_channels=in_chns,
            out_channels=class_num,
            init_filters=kwargs.get("init_filters", 32),
            kernel_size=kwargs.get("kernel_size", 3),
            deep_supervision=False,
        )
        return TupleOut(net)

    if net_type == "transunet":
        from comparison.networks.transunet3d import build_transunet3d
        net = build_transunet3d(
            in_channels=in_chns,
            num_classes=class_num,
            patch_size=kwargs.get("img_size", (96, 96, 96)),
        )
        # Returns a deep-supervision tuple; index 0 is the full-resolution head.
        return TupleOut(net, pick=lambda o: o[0])

    raise ValueError(
        f"Unknown net_type={net_type!r}. Choose from: "
        "unet_2d, unet_3d, vnet, unetr, unetpp, transunet, swinunetr, mednext."
    )
