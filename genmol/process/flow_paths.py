"""Interpolant paths for flow matching.

A path defines `x_t = path.interp(x_0, x_1, t)` and the target velocity for
training as `path.target_velocity(x_0, x_1, t)`. The base distribution
sample lives at `t=0` and the data sample at `t=1`.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
from torch import Tensor


def _broadcast_t(t: Tensor, ref: Tensor) -> Tensor:
    """Reshape `t` (shape (B,)) so it can broadcast against `ref`."""
    while t.dim() < ref.dim():
        t = t.unsqueeze(-1)
    return t


class InterpolantPath(ABC):
    @abstractmethod
    def interp(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor: ...
    @abstractmethod
    def target_velocity(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor: ...


class LinearPath(InterpolantPath):
    """Conditional OT (Lipman 2023) — straight-line interpolant.

    `x_t = (1-t) * x_0 + t * x_1`
    `v_target = x_1 - x_0` (constant over t for a given (x_0, x_1) pair)

    Easiest to reason about. The constant-target property can make training
    slow at intermediate t since the model has to map a t-varying input
    to a t-independent output.
    """

    def interp(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        t = _broadcast_t(t, x0)
        return (1 - t) * x0 + t * x1

    def target_velocity(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        return x1 - x0


class VPPath(InterpolantPath):
    """Variance-preserving cos/sin path (Albergo 2023 / common SDE-VP form).

    `x_t = cos(π t / 2) * x_0 + sin(π t / 2) * x_1`
    `v_target = (π/2) * (-sin(π t / 2) * x_0 + cos(π t / 2) * x_1)`

    Preserves marginal variance when `x_0, x_1` are unit-variance, which
    gives more stable training signal at intermediate t than the linear
    path. Recommended fallback when LinearPath under-trains.
    """

    HALF_PI = math.pi / 2

    def interp(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        t = _broadcast_t(t, x0)
        a = torch.cos(self.HALF_PI * t)
        b = torch.sin(self.HALF_PI * t)
        return a * x0 + b * x1

    def target_velocity(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        t = _broadcast_t(t, x0)
        da = -self.HALF_PI * torch.sin(self.HALF_PI * t)
        db = self.HALF_PI * torch.cos(self.HALF_PI * t)
        return da * x0 + db * x1


PATHS = {
    "linear": LinearPath,
    "vp": VPPath,
}
