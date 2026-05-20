"""Equivariance contract: rotating the input rotates the predicted coordinates."""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("torch_geometric", reason="EGNN test needs torch")  # actually only torch

from genmol.nn.egnn import EGNNBackbone


def _rand_rot():
    A = torch.randn(3, 3)
    Q, _ = torch.linalg.qr(A)
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def test_egnn_rotation_equivariance():
    torch.manual_seed(0)
    feat = 5
    net = EGNNBackbone(atom_feat_dim=feat, hidden_dim=32, n_layers=2, time_embed_dim=8)
    net.eval()
    B, N = 2, 6
    x = torch.randn(B, N, 3)
    h = torch.zeros(B, N, feat)
    h[..., 0] = 1.0
    t = torch.full((B,), 0.5)
    mask = torch.ones(B, N)

    pred_x1, pred_h1 = net(x, h, t, mask)

    R = _rand_rot()
    x_rot = x @ R
    pred_x2, pred_h2 = net(x_rot, h, t, mask)

    # Equivariance: pred_x2 should equal pred_x1 @ R (up to numerical error).
    expected = pred_x1 @ R
    assert torch.allclose(pred_x2, expected, atol=1e-4), (
        f"EGNN not equivariant under rotation. Max diff: "
        f"{(pred_x2 - expected).abs().max().item()}"
    )
    # Invariance of features
    assert torch.allclose(pred_h1, pred_h2, atol=1e-4), (
        f"EGNN features not invariant. Max diff: "
        f"{(pred_h1 - pred_h2).abs().max().item()}"
    )
