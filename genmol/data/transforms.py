"""Data transforms applied at __getitem__ time."""
from __future__ import annotations

import torch
from torch import Tensor

from genmol.utils.geometry import remove_mean


def random_rotation_3d(x: Tensor) -> Tensor:
    """Apply a random 3D rotation to coords. x: (N, 3) or (B, N, 3)."""
    A = torch.randn(3, 3)
    Q, _ = torch.linalg.qr(A)
    # Ensure positive determinant (proper rotation, not reflection)
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return x @ Q.to(x.dtype)


def center_of_mass_zero(x: Tensor, mask: Tensor) -> Tensor:
    """Remove the COM from coords given a mask."""
    if x.dim() == 2:
        x = x.unsqueeze(0)
        mask = mask.unsqueeze(0)
        return remove_mean(x, mask).squeeze(0)
    return remove_mean(x, mask)
