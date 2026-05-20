"""Geometric utilities shared across processes, backbones, and samplers."""
from __future__ import annotations

import torch
from torch import Tensor


def center_of_mass(x: Tensor, mask: Tensor) -> Tensor:
    """Returns per-sample center of mass.

    Args:
        x:    (B, N, 3)
        mask: (B, N)  1 = real atom, 0 = padding
    Returns:
        (B, 1, 3)
    """
    mask_f = mask.unsqueeze(-1).float()
    n_atoms = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
    com = (x * mask_f).sum(dim=1, keepdim=True) / n_atoms
    return com


def remove_mean(x: Tensor, mask: Tensor) -> Tensor:
    """COM-zero the coordinates of each sample (mask-aware)."""
    com = center_of_mass(x, mask)
    return (x - com) * mask.unsqueeze(-1).float()


def sample_com_zero_noise(shape: tuple[int, int, int], mask: Tensor, device) -> Tensor:
    """Sample Gaussian noise then project onto the COM-zero subspace.

    This is what EDM does to keep all sampled states translation-invariant.

    Args:
        shape: (B, N, 3)
        mask:  (B, N)
    Returns:
        (B, N, 3) — Gaussian noise with COM=0 per sample.
    """
    noise = torch.randn(shape, device=device)
    noise = noise * mask.unsqueeze(-1).float()
    return remove_mean(noise, mask)
