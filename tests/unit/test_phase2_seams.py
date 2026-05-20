"""M5 verifier: all Phase-2 seams import and have the expected contracts."""
from __future__ import annotations

import pytest

from genmol.rl import PolicyWrapper, RewardFn
from genmol.rl.ddpo import train_ddpo
from genmol.rl.dpo import train_dpo
from genmol.rl.flowgrpo import train_flowgrpo
from genmol.rl.preference_dataset import PreferenceDataset
from genmol.sample.guidance import compose_guidance, zero_guidance


def test_phase2_stubs_raise_not_implemented():
    with pytest.raises(NotImplementedError):
        train_ddpo(None, None)
    with pytest.raises(NotImplementedError):
        train_flowgrpo(None, None)
    with pytest.raises(NotImplementedError):
        train_dpo(None, None)


def test_policy_wrapper_is_abstract():
    with pytest.raises(TypeError):
        PolicyWrapper(sampler=None)


def test_reward_fn_is_abstract():
    with pytest.raises(TypeError):
        RewardFn()


def test_preference_dataset_is_abstract():
    with pytest.raises(TypeError):
        PreferenceDataset()


def test_zero_guidance_returns_zeros():
    import torch

    x = torch.randn(2, 5, 3)
    out = zero_guidance(x, t=0.5, cond=None)
    assert out.shape == x.shape
    assert (out == 0).all()


def test_compose_guidance_combines():
    import torch

    def g1(x, t, c):
        return torch.ones_like(x)

    def g2(x, t, c):
        return 2 * torch.ones_like(x)

    composed = compose_guidance([g1, g2], [0.5, 1.0])
    out = composed(torch.zeros(1, 2, 3), t=0.0, cond=None)
    assert (out == 2.5).all()
