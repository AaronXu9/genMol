"""Regression tests for the decoder.

History: prior to commit `43a6322`, the decoder was calling
`Chem.rdDetermineBonds.DetermineConnectivity` (a non-existent attribute) and
a bare `except Exception:` silently caught the `AttributeError`, returning
None for EVERY decoded molecule. This made the in-training `rdkit_eval`
callback report 0% validity for an entire 200k-step run before being noticed.

These tests would have caught the bug at first commit. Concretely they
require: given clean coordinates for known small molecules (methane,
water, ethanol), `decode_to_mol` must return a non-None RDKit Mol whose
canonical SMILES matches.
"""
from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from genmol.sample.decode import decode_batch, decode_to_mol
from genmol.utils.chem import QM9_ATOMIC_NUMBERS


def _ch4_coords() -> np.ndarray:
    """Tetrahedral methane coordinates, C-H bonds ~1.09 Å."""
    return np.array(
        [
            [0.0, 0.0, 0.0],          # C
            [1.09, 0.0, 0.0],         # H
            [-0.36, 1.028, 0.0],      # H
            [-0.36, -0.514, 0.890],   # H
            [-0.36, -0.514, -0.890],  # H
        ],
        dtype=np.float32,
    )


def _h2o_coords() -> np.ndarray:
    """Water O-H bonds ~0.96 Å at 104.5°."""
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [0.757, 0.586, 0.0],
            [-0.757, 0.586, 0.0],
        ],
        dtype=np.float32,
    )


def _qm9_index(atomic_number: int) -> int:
    return QM9_ATOMIC_NUMBERS.index(atomic_number)


def _argmax(atomic_numbers: list[int]) -> np.ndarray:
    return np.array([_qm9_index(z) for z in atomic_numbers])


def test_decode_methane_returns_valid_mol():
    mol = decode_to_mol(
        x=_ch4_coords(),
        h_argmax=_argmax([6, 1, 1, 1, 1]),
        mask=np.ones(5),
    )
    assert mol is not None, "decoder must return a non-None mol for clean methane"
    # Canonical SMILES of methane is 'C'.
    assert Chem.MolToSmiles(Chem.RemoveHs(mol)) == "C"


def test_decode_water_returns_valid_mol():
    mol = decode_to_mol(
        x=_h2o_coords(),
        h_argmax=_argmax([8, 1, 1]),
        mask=np.ones(3),
    )
    assert mol is not None
    assert Chem.MolToSmiles(Chem.RemoveHs(mol)) == "O"


def test_decode_too_few_atoms_returns_none():
    """Single atom is below the n_real >= 2 threshold."""
    mol = decode_to_mol(
        x=np.zeros((1, 3), dtype=np.float32),
        h_argmax=np.array([1]),
        mask=np.ones(1),
    )
    assert mol is None


def test_decode_batch_methane_count():
    """decode_batch returns one entry per batch element."""
    import torch

    x = torch.tensor(_ch4_coords()).unsqueeze(0).repeat(3, 1, 1)
    h = torch.zeros(3, 5, 5)
    # one-hot for atom indices [C, H, H, H, H]
    h[:, 0, _qm9_index(6)] = 1.0
    h[:, 1:, _qm9_index(1)] = 1.0
    mask = torch.ones(3, 5)
    mols = decode_batch(x, h, mask)
    assert len(mols) == 3
    assert all(m is not None for m in mols), \
        f"all three should decode, got {[m is not None for m in mols]}"


def test_decode_propagates_attribute_errors():
    """If a future change breaks an RDKit import path, the decoder must
    crash loudly (not silently return None). This is the regression that
    bit us — see commit 43a6322 and the module docstring of this test file.
    """
    from unittest.mock import patch

    import genmol.sample.decode as decode_mod

    class _Broken:
        def DetermineBonds(self, *a, **kw):
            raise AttributeError("simulated future API breakage")

    with patch.object(decode_mod, "rdDetermineBonds", _Broken()):
        with pytest.raises(AttributeError, match="simulated"):
            decode_to_mol(
                x=_ch4_coords(),
                h_argmax=_argmax([6, 1, 1, 1, 1]),
                mask=np.ones(5),
            )
