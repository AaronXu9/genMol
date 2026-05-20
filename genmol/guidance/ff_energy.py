"""Phase-2: force-field (MMFF/UFF) energy gradient as a guidance signal."""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from genmol.sample.sampler import GuidanceFn


class FFEnergyGuidance:
    """Computes ∇x of a force-field energy through RDKit and returns it as
    a guidance term added to the score/velocity at sampling time."""

    def __init__(self, kind: str = "uff"):
        if kind not in {"mmff", "uff"}:
            raise ValueError(kind)
        self.kind = kind

    def __call__(self, x: Tensor, t: float, cond: Optional[dict]) -> Tensor:
        raise NotImplementedError(
            "Phase 2: convert x to RDKit conformer, run FF, finite-diff to get ∇x."
        )
