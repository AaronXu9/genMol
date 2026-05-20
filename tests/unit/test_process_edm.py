"""EDM forward-corruption marginal sanity tests."""
from __future__ import annotations

import torch

from genmol.data.base import ProcessBatch
from genmol.process.edm import EDMProcess
from genmol.process.schedules import PolynomialSchedule


def test_polynomial_schedule_boundaries():
    sched = PolynomialSchedule()
    t0 = torch.tensor([0.0])
    t1 = torch.tensor([1.0])
    assert sched.alpha(t0) > 0.99
    assert sched.alpha(t1) < 0.01


def test_edm_corrupt_preserves_mask():
    proc = EDMProcess(schedule="polynomial")
    x = torch.randn(2, 4, 3)
    h = torch.zeros(2, 4, 5)
    h[..., 0] = 1.0
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.float32)
    batch = ProcessBatch(x=x, h=h, mask=mask)
    t = torch.tensor([0.2, 0.8])
    corrupted = proc.corrupt(batch, t)
    # Padded positions should not be touched (well, mask is preserved).
    assert torch.equal(corrupted.mask, mask)
    # The aux dict carries eps for the loss.
    assert "eps_x" in corrupted.aux
    assert corrupted.aux["eps_x"].shape == x.shape
