"""Sampler must accept guidance_fn=None AND a trivial zero-guidance callable
without changing outputs (modulo float noise). This keeps the Phase-2 seam alive.
"""
from __future__ import annotations

import torch
from torch import nn

from genmol.nn.backbone import Backbone
from genmol.process.edm import EDMProcess
from genmol.sample.guidance import zero_guidance
from genmol.sample.sampler_impl import DiffusionSampler


class TinyBackbone(Backbone):
    prediction_type = "noise"

    def __init__(self):
        super().__init__()
        self.atom_feat_dim = 5
        self.dummy = nn.Linear(1, 1)  # ensures .parameters() is non-empty

    def forward(self, x, h, t, mask, cond=None):
        return torch.zeros_like(x), torch.zeros_like(h)


def test_guidance_none_vs_zero_guidance_match():
    torch.manual_seed(0)
    model = TinyBackbone()
    proc = EDMProcess(schedule="polynomial")
    sampler = DiffusionSampler(model, proc, atom_feat_dim=5, num_steps=4)

    out1 = sampler.sample(n_samples=2, n_atoms=4, cond=None, guidance_fn=None, seed=42)
    out2 = sampler.sample(
        n_samples=2, n_atoms=4, cond=None,
        guidance_fn=zero_guidance, guidance_scale=1.0, seed=42,
    )
    assert torch.allclose(out1["x"], out2["x"], atol=1e-5)
    assert torch.allclose(out1["h"], out2["h"], atol=1e-5)
