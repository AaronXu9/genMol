"""Noise schedules for diffusion processes.

Convention: schedules return `alpha_t` and `sigma_t` such that
    x_t = alpha_t * x_0 + sigma_t * eps,   eps ~ N(0, I)
EDM uses a variance-preserving polynomial schedule (alpha^2 + sigma^2 = 1).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class NoiseSchedule(ABC):
    @abstractmethod
    def alpha(self, t: Tensor) -> Tensor: ...
    @abstractmethod
    def sigma(self, t: Tensor) -> Tensor: ...

    def snr(self, t: Tensor) -> Tensor:
        a = self.alpha(t)
        s = self.sigma(t)
        return (a * a) / (s * s).clamp(min=1e-12)


class PolynomialSchedule(NoiseSchedule):
    """EDM's polynomial schedule (Hoogeboom 2022 Eq. 10–11)."""

    def __init__(self, power: float = 2.0, s: float = 1e-5):
        self.power = power
        self.s = s

    def alpha2(self, t: Tensor) -> Tensor:
        return (1.0 - 2.0 * self.s) * (1.0 - t.float() ** self.power) ** 2 + self.s

    def alpha(self, t: Tensor) -> Tensor:
        return self.alpha2(t).clamp(min=1e-12).sqrt()

    def sigma(self, t: Tensor) -> Tensor:
        return (1.0 - self.alpha2(t)).clamp(min=1e-12).sqrt()


class CosineSchedule(NoiseSchedule):
    """Cosine schedule (Nichol & Dhariwal 2021)."""

    def __init__(self, s: float = 0.008):
        self.s = s

    def alpha(self, t: Tensor) -> Tensor:
        return torch.cos(((t + self.s) / (1 + self.s)) * (torch.pi / 2)).clamp(min=1e-12)

    def sigma(self, t: Tensor) -> Tensor:
        a = self.alpha(t)
        return (1.0 - a * a).clamp(min=1e-12).sqrt()


SCHEDULES = {
    "polynomial": PolynomialSchedule,
    "cosine": CosineSchedule,
}
