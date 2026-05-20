"""Padded-batch collator producing ProcessBatch-ready tensors.

We use a padded tensor (B, N_max, ...) representation rather than PyG Batch
because our backbone is a dense EGNN over fully-connected per-sample graphs.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor


def pad_collate(items: Sequence[dict]) -> dict:
    """Items: list of dicts with keys 'x', 'h', 'mask'. Returns padded batch.

    Args:
        items: list of length B, each dict with
               'x': (n_i, 3), 'h': (n_i, F), 'mask': (n_i,) (all 1s).
    Returns:
        Dict with 'x': (B, N_max, 3), 'h': (B, N_max, F), 'mask': (B, N_max).
    """
    B = len(items)
    n_max = max(int(item["mask"].shape[0]) for item in items)
    feat_dim = int(items[0]["h"].shape[-1])

    x = torch.zeros(B, n_max, 3)
    h = torch.zeros(B, n_max, feat_dim)
    mask = torch.zeros(B, n_max)
    for b, item in enumerate(items):
        n = int(item["mask"].shape[0])
        x[b, :n] = item["x"]
        h[b, :n] = item["h"]
        mask[b, :n] = item["mask"].float()
    return {"x": x, "h": h, "mask": mask}
