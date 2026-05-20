"""Optimizer/scheduler builders. Mostly unused — BaseGenLitModule has its own."""
from __future__ import annotations

import torch
from torch import nn


def build_adamw(params, lr: float = 1e-4, weight_decay: float = 1e-12):
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
