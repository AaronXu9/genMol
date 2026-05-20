"""Smoke: from a freshly initialized (untrained) model, sample 5 molecules
and confirm the decoder runs end-to-end (likely returns mostly None mols)."""
from __future__ import annotations

import torch
from torch import nn

from genmol.nn.backbone import Backbone
from genmol.process.edm import EDMProcess
from genmol.sample.decode import decode_batch
from genmol.sample.sampler_impl import DiffusionSampler


class _DummyBB(Backbone):
    prediction_type = "noise"

    def __init__(self):
        super().__init__()
        self.atom_feat_dim = 5
        self.fc = nn.Linear(1, 1)

    def forward(self, x, h, t, mask, cond=None):
        return torch.zeros_like(x), torch.zeros_like(h)


def test_sample_and_decode_runs():
    proc = EDMProcess(schedule="polynomial")
    sampler = DiffusionSampler(_DummyBB(), proc, atom_feat_dim=5, num_steps=5)
    out = sampler.sample(n_samples=5, n_atoms=8, seed=0)
    assert out["x"].shape == (5, 8, 3)
    assert out["h"].shape == (5, 8, 5)
    mols = decode_batch(out["x"], out["h"], out["mask"])
    assert len(mols) == 5
