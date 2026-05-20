"""Composer for guidance functions (Phase-2 seam)."""
from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor

from genmol.sample.sampler import GuidanceFn


def zero_guidance(x: Tensor, t: float, cond: Optional[dict]) -> Tensor:
    """No-op guidance: returns zeros with the same shape as x."""
    return torch.zeros_like(x)


def compose_guidance(fns: Sequence[GuidanceFn], weights: Sequence[float]) -> GuidanceFn:
    """Linearly combine multiple guidance functions."""
    if len(fns) != len(weights):
        raise ValueError("len(fns) must equal len(weights)")

    def composed(x: Tensor, t: float, cond: Optional[dict]) -> Tensor:
        out = torch.zeros_like(x)
        for f, w in zip(fns, weights):
            out = out + w * f(x, t, cond)
        return out

    return composed
