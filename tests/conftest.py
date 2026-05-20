"""Shared pytest fixtures."""
from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True)
def _torch_deterministic():
    torch.manual_seed(0)
    yield


@pytest.fixture
def small_batch():
    """A tiny batch suitable for fast unit tests."""
    B, N, F = 2, 5, 5
    x = torch.randn(B, N, 3)
    h = torch.zeros(B, N, F)
    h[..., 0] = 1.0  # all-hydrogen
    mask = torch.ones(B, N)
    return {"x": x, "h": h, "mask": mask}
