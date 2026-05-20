"""Small neural building blocks (FiLM, gated MLPs)."""
from __future__ import annotations

from torch import Tensor, nn


class FiLM(nn.Module):
    """Feature-wise linear modulation conditioned on a per-sample vector."""

    def __init__(self, cond_dim: int, feature_dim: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * feature_dim)

    def forward(self, h: Tensor, c: Tensor) -> Tensor:
        """h: (B, N, F), c: (B, C). Broadcast scale+shift over N."""
        gamma_beta = self.proj(c).unsqueeze(1)  # (B, 1, 2F)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return h * (1 + gamma) + beta
