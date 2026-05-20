"""Featurization helpers: atom one-hots, mask construction."""
from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor


def atom_types_to_onehot(z: Tensor, atom_types: Sequence[int]) -> Tensor:
    """Map atomic numbers to a one-hot encoding given the dataset's atom set.

    Args:
        z:          (N,) int tensor of atomic numbers.
        atom_types: list of atomic numbers in canonical order for this dataset.
    Returns:
        (N, len(atom_types)) float one-hot.
    """
    out = torch.zeros(z.shape[0], len(atom_types), dtype=torch.float32)
    for i, an in enumerate(atom_types):
        out[:, i] = (z == an).float()
    return out
