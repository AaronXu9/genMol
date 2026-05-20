"""Padded collate produces correctly-shaped batches."""
from __future__ import annotations

import torch

from genmol.data.collate import pad_collate


def test_pad_collate_shapes():
    items = [
        {"x": torch.randn(3, 3), "h": torch.zeros(3, 4), "mask": torch.ones(3)},
        {"x": torch.randn(5, 3), "h": torch.zeros(5, 4), "mask": torch.ones(5)},
    ]
    out = pad_collate(items)
    assert out["x"].shape == (2, 5, 3)
    assert out["h"].shape == (2, 5, 4)
    assert out["mask"].shape == (2, 5)
    assert out["mask"][0, 3:].sum() == 0
    assert out["mask"][0, :3].sum() == 3
    assert out["mask"][1, :].sum() == 5
