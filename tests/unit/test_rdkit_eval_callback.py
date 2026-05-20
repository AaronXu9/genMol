"""Tests for `RDKitEvalCallback` zero-validity warning logic.

The warning is the watchdog that would have surfaced the decoder bug
(43a6322) within a few thousand steps instead of 200k.
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock

from genmol.train.callbacks.rdkit_eval import RDKitEvalCallback


def test_zero_validity_warns_after_threshold(caplog):
    cb = RDKitEvalCallback(
        zero_validity_warn_after=3,
        zero_validity_warmup_steps=1000,
    )
    # Below warmup: should not warn even after many zero evals.
    with caplog.at_level(logging.WARNING, logger="genmol.callbacks.rdkit_eval"):
        for _ in range(10):
            cb._maybe_warn_zero_validity(step=500, validity=0.0)
    assert not any("validity has been 0.0" in m for m in caplog.messages)

    # Past warmup: should warn on the 3rd consecutive zero.
    caplog.clear()
    cb._consecutive_zero = 0  # reset
    with caplog.at_level(logging.WARNING, logger="genmol.callbacks.rdkit_eval"):
        for _ in range(3):
            cb._maybe_warn_zero_validity(step=2000, validity=0.0)
    matches = [m for m in caplog.messages if "validity has been 0.0" in m]
    assert len(matches) >= 1, f"expected warning; got messages: {caplog.messages}"


def test_zero_streak_resets_on_nonzero_validity(caplog):
    cb = RDKitEvalCallback(
        zero_validity_warn_after=2,
        zero_validity_warmup_steps=0,
    )
    cb._maybe_warn_zero_validity(step=1000, validity=0.0)
    assert cb._consecutive_zero == 1
    # A single non-zero observation resets the counter.
    cb._maybe_warn_zero_validity(step=1100, validity=0.3)
    assert cb._consecutive_zero == 0
    # And the warning doesn't fire on the next zero alone.
    with caplog.at_level(logging.WARNING, logger="genmol.callbacks.rdkit_eval"):
        cb._maybe_warn_zero_validity(step=1200, validity=0.0)
    assert not any("validity has been 0.0" in m for m in caplog.messages)


def test_zero_validity_warns_at_exactly_threshold(caplog):
    cb = RDKitEvalCallback(
        zero_validity_warn_after=5,
        zero_validity_warmup_steps=0,
    )
    with caplog.at_level(logging.WARNING, logger="genmol.callbacks.rdkit_eval"):
        for i in range(5):
            cb._maybe_warn_zero_validity(step=1000 + i, validity=0.0)
    # Should have warned exactly once (on the 5th call).
    n_warnings = sum("validity has been 0.0" in m for m in caplog.messages)
    assert n_warnings >= 1
