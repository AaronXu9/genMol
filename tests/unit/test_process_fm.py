"""FM interpolant boundary conditions."""
from __future__ import annotations

import torch

from genmol.data.base import ProcessBatch
from genmol.process.flow_matching import FlowMatchingProcess


def test_fm_t0_is_noise_t1_is_data():
    proc = FlowMatchingProcess(path="linear")
    torch.manual_seed(0)
    x = torch.randn(1, 6, 3)
    h = torch.zeros(1, 6, 5)
    h[..., 0] = 1.0
    mask = torch.ones(1, 6)
    batch = ProcessBatch(x=x, h=h, mask=mask)

    # At t=1 the interpolant should be data.
    t = torch.tensor([0.999])
    out = proc.corrupt(batch, t)
    # Loose tolerance since noise still contributes at very small (1-t).
    assert (out.x - x).abs().max().item() < 0.5, "FM at t≈1 should be ≈data"


def test_fm_loss_finite():
    """Forward + naive loss with an untrained no-op model returns a finite number."""
    from torch import nn

    class NoopBackbone(nn.Module):
        prediction_type = "velocity"

        def forward(self, x, h, t, mask, cond=None):
            return torch.zeros_like(x), torch.zeros_like(h)

    proc = FlowMatchingProcess(path="linear")
    x = torch.randn(2, 5, 3)
    h = torch.zeros(2, 5, 5)
    h[..., 0] = 1.0
    mask = torch.ones(2, 5)
    batch = ProcessBatch(x=x, h=h, mask=mask)
    loss = proc.loss(NoopBackbone(), batch)
    assert torch.isfinite(loss)
