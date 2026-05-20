"""Interpolant paths for flow matching.

A path defines x_t = path.interp(x_0, x_1, t), and the target velocity for
training as path.target_velocity(x_0, x_1, t).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor


class InterpolantPath(ABC):
    @abstractmethod
    def interp(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor: ...
    @abstractmethod
    def target_velocity(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor: ...


class LinearPath(InterpolantPath):
    """Conditional OT (Lipman 2023) — straight-line interpolant."""

    def interp(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        # t is (B,); broadcast over (N, 3)
        while t.dim() < x0.dim():
            t = t.unsqueeze(-1)
        return (1 - t) * x0 + t * x1

    def target_velocity(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        return x1 - x0


PATHS = {"linear": LinearPath}
