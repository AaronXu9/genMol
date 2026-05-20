"""Smoke: the package imports cleanly without errors."""
from __future__ import annotations


def test_package_imports():
    import genmol
    import genmol.data
    import genmol.data.qm9
    import genmol.eval.metrics_2d
    import genmol.models.base_lit
    import genmol.nn.egnn
    import genmol.oracle.base
    import genmol.oracle.rdkit_oracle
    import genmol.process.edm
    import genmol.process.flow_matching
    import genmol.rl  # Phase-2 stubs
    import genmol.sample.sampler
    import genmol.sample.sampler_impl

    assert genmol.__version__
