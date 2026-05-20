"""Backbone ABC: an equivariant network that predicts a tensor whose
semantics are determined by the caller (Process). The backbone knows nothing
about diffusion vs flow matching.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from torch import Tensor, nn


class Backbone(nn.Module, ABC):
    """E(n)-equivariant network for joint (x, h) prediction.

    Subclass MUST set `prediction_type` at construction. This string is
    informational — Process consumers know what the network predicts because
    they configured it; the attribute exists to enable runtime sanity checks
    and to make config errors loud.
    """

    prediction_type: str  # 'noise' | 'velocity' | 'x0'

    @abstractmethod
    def forward(
        self,
        x: Tensor,                          # (B, N, 3)
        h: Tensor,                          # (B, N, F)
        t: Tensor,                          # (B,) in [0, 1]
        mask: Tensor,                       # (B, N)
        cond: Optional[dict] = None,        # pocket conditioning for SBDD
    ) -> tuple[Tensor, Tensor]:
        """Returns (pred_x, pred_h) with same shapes as inputs (modulo the
        feature dim for h, which may differ if predicting class logits).
        Semantics depend on `self.prediction_type`."""
