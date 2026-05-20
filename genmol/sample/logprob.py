"""Log-prob with gradient flow for Phase-2 RL.

For diffusion (DDIM step): sum of Gaussian log-probs along the reverse trajectory.
For Flow Matching (ODE): for FlowGRPO we use the stochastic SDE form;
deterministic ODE has zero on-policy gradient.

CRITICAL: this module MUST NOT use `torch.no_grad()` anywhere. The point of
Phase-2 RL is gradient flow through log_prob; wrapping it in no_grad silently
breaks the entire RL pipeline.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor


def gaussian_log_prob(x: Tensor, mean: Tensor, std: Tensor, mask: Tensor) -> Tensor:
    """Per-sample Gaussian log-prob, masked.

    Args:
        x:    (B, N, D)
        mean: (B, N, D) — typically the deterministic-step prediction.
        std:  (B, N, D) or scalar broadcast — stochastic-step noise scale.
        mask: (B, N)
    Returns:
        (B,) — per-sample log-prob, summed over all real atom dims.
    """
    if not isinstance(std, Tensor) or std.dim() == 0:
        std = torch.full_like(mean, float(std))
    var = std * std
    var = var.clamp(min=1e-12)
    log_prob_per_dim = -0.5 * ((x - mean) ** 2 / var + var.log() + math.log(2 * math.pi))
    m = mask.unsqueeze(-1).float()
    return (log_prob_per_dim * m).sum(dim=(-2, -1))
