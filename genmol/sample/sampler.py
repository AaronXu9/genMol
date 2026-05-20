"""Sampler ABC: inference layer with first-class hooks for Phase-2.

Two non-negotiable seams baked in from day one:
- `guidance_fn` — for physics-guided sampling.
- `return_log_prob` — for DDPO/FlowGRPO (requires `requires_grad=True` output).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional, Union

from torch import Tensor

from genmol.data.base import ProcessBatch

# A guidance fn maps (x_t, t, cond) -> a gradient term added to the score/velocity.
# Must NOT wrap with torch.no_grad(); Phase-2 RL relies on gradient flow through it.
GuidanceFn = Callable[[Tensor, float, Optional[dict]], Tensor]


class Sampler(ABC):
    """Wraps a (process, model) pair and produces samples."""

    @abstractmethod
    def sample(
        self,
        n_samples: int,
        n_atoms: Union[int, Tensor],
        cond: Optional[dict] = None,
        guidance_fn: Optional[GuidanceFn] = None,
        guidance_scale: float = 0.0,
        return_log_prob: bool = False,
        return_trajectory: bool = False,
        seed: Optional[int] = None,
        temperature: float = 1.0,
    ) -> dict:
        """Deterministic sample (ODE for FM, DDIM-like for diffusion).

        Returns dict with:
            'x':           (B, N, 3)
            'h':           (B, N, F)
            'mask':        (B, N)
            'log_prob':    (B,) WITH requires_grad=True, if return_log_prob.
            'trajectory':  list[ProcessBatch], if return_trajectory.
        """

    @abstractmethod
    def stochastic_sample(
        self,
        n_samples: int,
        n_atoms: Union[int, Tensor],
        cond: Optional[dict] = None,
        guidance_fn: Optional[GuidanceFn] = None,
        guidance_scale: float = 0.0,
        return_log_prob: bool = False,
        return_trajectory: bool = False,
        seed: Optional[int] = None,
        temperature: float = 1.0,
    ) -> dict:
        """Stochastic counterpart (SDE for FM, ancestral DDPM for diffusion).

        REQUIRED for DDPO/FlowGRPO: deterministic FM has zero on-policy
        gradient, so RL fine-tuning needs the SDE form. See `references/copilot-tools.md`
        in the FlowGRPO paper for the ODE↔SDE conversion.
        """
