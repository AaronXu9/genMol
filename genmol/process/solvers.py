"""ODE / SDE solvers for sampling. Process-agnostic; takes a `drift_fn`."""
from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor


def euler_step(x: Tensor, drift: Tensor, dt: Tensor) -> Tensor:
    return x + drift * dt


def heun_step(
    x: Tensor,
    t: Tensor,
    dt: Tensor,
    drift_fn: Callable[[Tensor, Tensor], Tensor],
) -> Tensor:
    """Heun's method (2nd order). drift_fn(x, t) -> Tensor."""
    k1 = drift_fn(x, t)
    x_euler = x + k1 * dt
    k2 = drift_fn(x_euler, t + dt)
    return x + 0.5 * (k1 + k2) * dt


SOLVERS = {"euler": "euler", "heun": "heun"}
