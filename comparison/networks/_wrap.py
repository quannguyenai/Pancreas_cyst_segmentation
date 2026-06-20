"""_wrap.py — output-contract adapter for the comparison harness.

The harness has two call sites with different output expectations:
  * ``train.py``           does ``outputs = model(x)``   (wants a single logits tensor)
  * ``utils/test_3d_patch`` does ``y1, _ = model(x)``     (wants a 2-tuple)

Different baseline networks return different things (single tensor, a
``(logits, features)`` tuple, or a list of deep-supervision heads). ``TupleOut``
normalizes any of them to a uniform ``(logits, logits)`` 2-tuple so both call
sites work unchanged. ``train.py`` additionally unwraps tuples (see its forward
guard), so the duplicated second element is simply ignored there.
"""

from __future__ import annotations

import torch.nn as nn


class TupleOut(nn.Module):
    """Wrap a segmentation net so ``forward`` always returns ``(logits, logits)``.

    Parameters
    ----------
    net : nn.Module
        The wrapped network.
    pick : callable, optional
        Selects the logits tensor from the wrapped net's raw output. Use for
        deep-supervision / multi-head nets, e.g. ``pick=lambda o: o[0]``. If not
        given, a list/tuple output defaults to its first element and a tensor is
        passed through unchanged.
    """

    def __init__(self, net: nn.Module, pick=None):
        super().__init__()
        self.net = net
        self.pick = pick

    def forward(self, x):
        out = self.net(x)
        if self.pick is not None:
            logits = self.pick(out)
        elif isinstance(out, (list, tuple)):
            logits = out[0]
        else:
            logits = out
        return logits, logits
