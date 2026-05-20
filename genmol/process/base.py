"""Process ABC: the forward (corruption) and reverse (generation) stochastic
process. THIS is where EDM and Flow Matching differ; the rest of the stack
(backbone, data, sampler) is shared.

Critical invariants:
- The backbone is a pure prediction module. NO loss math lives there.
- A Process is responsible for: sampling t, corrupting a clean batch, computing
  the training loss, taking a reverse step, and initialising noise/base samples.
- The Process MUST accept an optional `guidance_fn` in `step(...)` — this is
  the Phase-2 seam for physics-guided sampling. Even when unused, the parameter
  must be plumbed through so the seam stays alive and tested.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import torch
from torch import Tensor

from genmol.data.base import ProcessBatch

if TYPE_CHECKING:
    from genmol.nn.backbone import Backbone
    from genmol.sample.sampler import GuidanceFn


class Process(ABC):
    """Joint denoising of (x, h) — coordinates and atom features."""

    @property
    @abstractmethod
    def prediction_type(self) -> str:
        """One of 'noise' | 'velocity' | 'x0'.

        The backbone's output head shape and the loss formulation depend on this.
        EDM uses 'noise'; standard Flow Matching uses 'velocity'.
        """

    @abstractmethod
    def sample_t(self, batch_size: int, device: torch.device) -> Tensor:
        """Sample training time-steps in [0, 1]. Returns (B,)."""

    @abstractmethod
    def corrupt(self, batch: ProcessBatch, t: Tensor) -> ProcessBatch:
        """Forward corruption. Returns a new `ProcessBatch` whose `x`, `h` are
        the corrupted state at time `t`, and whose `aux` dict contains whatever
        the loss needs (noise sample, target velocity, etc.)."""

    @abstractmethod
    def loss(self, backbone: "Backbone", batch: ProcessBatch) -> Tensor:
        """End-to-end training loss for a clean batch. Internally:
        1. sample t,
        2. call `corrupt(batch, t)`,
        3. call `backbone.forward(...)`,
        4. compare to the target stored in `corrupted.aux`,
        5. return a scalar loss."""

    @abstractmethod
    def step(
        self,
        backbone: "Backbone",
        x_t: ProcessBatch,
        t: Tensor,
        dt: Tensor,
        guidance_fn: Optional["GuidanceFn"] = None,
        guidance_scale: float = 0.0,
    ) -> ProcessBatch:
        """One reverse step. `guidance_fn` is the Phase-2 hook for physics-
        guided sampling: it receives (x_t, t, cond) and returns a gradient term
        that is added to the score / velocity prediction with weight `guidance_scale`."""

    @abstractmethod
    def init_noise(
        self,
        n_atoms: Tensor,            # (B,) int — number of real atoms per sample
        max_atoms: int,
        feature_dim: int,
        cond: Optional[dict],
        device: torch.device,
    ) -> ProcessBatch:
        """Draw the initial state for sampling (x_T for diffusion, x_0 ~ base
        for FM). Coordinates are COM-zeroed."""

    @abstractmethod
    def t_schedule(self, n_steps: int, device: torch.device) -> Tensor:
        """Sampling timestep schedule. Returns a (n_steps+1,) tensor from t=1 → t=0
        for diffusion, or t=0 → t=1 for flow matching."""
