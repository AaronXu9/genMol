"""Sinusoidal time/sigma embedding shared by EDM and FM backbones."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class SinusoidalTimeEmbedding(nn.Module):
    """Fourier features for a scalar time/sigma in [0, 1]."""

    def __init__(self, dim: int, max_period: float = 10_000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: Tensor) -> Tensor:
        """t: (B,) in [0, 1]. Returns (B, dim)."""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t.float()[:, None] * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
