"""The Phase-2 critical contract: stochastic_sample with return_log_prob=True
must return a Tensor whose gradient flows back to backbone parameters.

If this test ever passes only because of `torch.no_grad()` somewhere, Phase-2
FlowGRPO will silently fail. Do not weaken this test.
"""
from __future__ import annotations

import torch
from torch import nn

from genmol.nn.backbone import Backbone
from genmol.process.flow_matching import FlowMatchingProcess
from genmol.sample.sampler_impl import FlowSampler


class TinyParamBackbone(Backbone):
    prediction_type = "velocity"

    def __init__(self, feat=5):
        super().__init__()
        self.atom_feat_dim = feat
        self.fc_x = nn.Linear(3, 3)
        self.fc_h = nn.Linear(feat, feat)

    def forward(self, x, h, t, mask, cond=None):
        return self.fc_x(x) * mask.unsqueeze(-1), self.fc_h(h) * mask.unsqueeze(-1)


def test_logprob_has_grad_and_flows_back():
    torch.manual_seed(0)
    model = TinyParamBackbone(feat=5)
    proc = FlowMatchingProcess(path="linear")
    sampler = FlowSampler(model, proc, atom_feat_dim=5, num_steps=4)

    out = sampler.stochastic_sample(
        n_samples=2, n_atoms=4, return_log_prob=True, seed=0,
    )
    assert out["log_prob"] is not None
    # Gradient must flow back to model parameters.
    loss = -out["log_prob"].mean()
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0, "no parameters received gradients"
    assert any(g > 0 for g in grad_norms), "all gradients are zero — check no_grad usage"
