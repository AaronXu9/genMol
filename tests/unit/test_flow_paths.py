"""Unit tests for flow-matching interpolant paths."""
from __future__ import annotations

import math

import pytest
import torch

from genmol.process.flow_paths import PATHS, LinearPath, VPPath


@pytest.fixture
def x_pair():
    torch.manual_seed(0)
    x0 = torch.randn(3, 5, 3)
    x1 = torch.randn(3, 5, 3)
    return x0, x1


# === LinearPath ===

def test_linear_boundaries(x_pair):
    x0, x1 = x_pair
    p = LinearPath()
    assert torch.allclose(p.interp(x0, x1, torch.zeros(3)), x0)
    assert torch.allclose(p.interp(x0, x1, torch.ones(3)), x1)


def test_linear_velocity_is_constant(x_pair):
    x0, x1 = x_pair
    p = LinearPath()
    v_at_0 = p.target_velocity(x0, x1, torch.zeros(3))
    v_at_1 = p.target_velocity(x0, x1, torch.ones(3))
    v_at_half = p.target_velocity(x0, x1, torch.full((3,), 0.5))
    expected = x1 - x0
    assert torch.allclose(v_at_0, expected)
    assert torch.allclose(v_at_1, expected)
    assert torch.allclose(v_at_half, expected)


# === VPPath ===

def test_vp_boundaries(x_pair):
    x0, x1 = x_pair
    p = VPPath()
    # At t=0: a=1, b=0 → returns x0.
    assert torch.allclose(p.interp(x0, x1, torch.zeros(3)), x0, atol=1e-6)
    # At t=1: a=0, b=1 → returns x1.
    assert torch.allclose(p.interp(x0, x1, torch.ones(3)), x1, atol=1e-6)


def test_vp_variance_preservation():
    """If x_0, x_1 ~ N(0, I) independently, then x_t = cos(πt/2)x_0 + sin(πt/2)x_1
    is also marginally N(0, I) — that's the VP property.
    """
    torch.manual_seed(42)
    N = 50_000
    x0 = torch.randn(N, 3)
    x1 = torch.randn(N, 3)
    p = VPPath()
    for t_val in (0.25, 0.5, 0.75):
        xt = p.interp(x0, x1, torch.full((N,), t_val))
        var = xt.var(dim=0).mean().item()
        assert abs(var - 1.0) < 0.02, f"VP path failed variance preservation at t={t_val}: var={var}"


def test_vp_velocity_matches_derivative(x_pair):
    """The closed-form velocity should match a numerical d/dt of interp."""
    x0, x1 = x_pair
    p = VPPath()
    t_val = 0.4
    eps = 1e-4
    t_plus = torch.full((3,), t_val + eps)
    t_minus = torch.full((3,), t_val - eps)
    numerical = (p.interp(x0, x1, t_plus) - p.interp(x0, x1, t_minus)) / (2 * eps)
    closed = p.target_velocity(x0, x1, torch.full((3,), t_val))
    # 5e-3 tolerance because central-difference at eps=1e-4 in fp32 has
    # ~1e-3 round-off error (machine_eps / eps); this is checking the
    # ANALYTICAL derivative matches, not asking for full precision.
    assert torch.allclose(numerical, closed, atol=5e-3), (
        f"VP velocity mismatch: max diff {(numerical - closed).abs().max()}"
    )


# === Registry ===

def test_paths_registry_has_linear_and_vp():
    assert "linear" in PATHS
    assert "vp" in PATHS
