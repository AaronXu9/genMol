"""Phase-2 seam: RewardFn ABC + composite reward.

A RewardFn maps generated samples (and optional conditioning) to a per-sample
scalar reward. Typically composed of one or more `Oracle`s.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor

from genmol.oracle.base import Oracle


class RewardFn(ABC):
    @abstractmethod
    def __call__(self, samples: dict, cond: Optional[dict] = None) -> Tensor:
        """Returns (B,) reward tensor. May aggregate multiple Oracles."""


class WeightedOracleReward(RewardFn):
    """A simple weighted sum of oracle scores.

    Phase-2 work will likely subclass this to handle:
    - Normalization (z-score per oracle, running mean)
    - Constraint penalties (validity gate)
    - Clipping
    """

    def __init__(self, oracles: list[Oracle], weights: list[float]):
        if len(oracles) != len(weights):
            raise ValueError("len(oracles) must equal len(weights)")
        self.oracles = oracles
        self.weights = weights

    def __call__(self, samples: dict, cond: Optional[dict] = None) -> Tensor:
        raise NotImplementedError(
            "Phase 2: implement decode(samples) → mols, oracle call, weighted sum"
        )
