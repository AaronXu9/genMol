"""Phase-2 seam: PolicyWrapper.

Wraps a trained Sampler as an RL policy. The Phase-1 Sampler exposes
`stochastic_sample` and `return_log_prob=True` — that is enough for DDPO,
FlowGRPO, PPO-style rollouts to be implemented HERE without touching the
Phase-1 generator.

DO NOT implement the RL trainer here. This file only defines the interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from torch import Tensor

from genmol.sample.sampler import Sampler


class PolicyWrapper(ABC):
    def __init__(self, sampler: Sampler, ref_sampler: Optional[Sampler] = None):
        """Args:
            sampler:     the trainable policy (its parameters get gradients).
            ref_sampler: a frozen reference (for KL constraint / DPO / PPO ratio).
                         None means no reference (vanilla policy gradient).
        """
        self.sampler = sampler
        self.ref_sampler = ref_sampler

    @abstractmethod
    def rollout(self, batch_size: int, cond: Optional[dict] = None) -> dict:
        """Generate samples USING `stochastic_sample` so the rollout is
        on-policy. Must return dict including 'log_prob' (Tensor with grad)
        and 'trajectory' (list[ProcessBatch])."""

    @abstractmethod
    def log_prob(self, samples: dict, cond: Optional[dict] = None) -> Tensor:
        """Re-evaluate log-prob of given samples under current parameters
        (for PPO importance ratio / GRPO advantage computation)."""
