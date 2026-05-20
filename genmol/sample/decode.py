"""Decode generator output tensors into RDKit Mol objects.

For unconditional 3D generators we have continuous coordinates + atom-type
logits. To produce a valid SMILES we need:
1. argmax over atom-type logits (or one-hot continuous).
2. Construct an RDKit Mol with atomic numbers + 3D coords.
3. Run OpenBabel-style bond perception (RDKit's `DetermineBonds` works for
   QM9-sized molecules; for drug-like molecules we use openbabel as fallback).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from torch import Tensor

from genmol.utils.chem import QM9_ATOMIC_NUMBERS


def decode_to_mol(
    x: np.ndarray,            # (N, 3)
    h_argmax: np.ndarray,     # (N,) atom-type indices
    mask: np.ndarray,         # (N,) 1=real
    atomic_numbers: Sequence[int] = QM9_ATOMIC_NUMBERS,
) -> Optional[Chem.Mol]:
    """Build an RDKit Mol from (coords, atom-type indices, mask). Returns None
    on failure."""
    n_real = int(mask.sum())
    if n_real < 2:
        return None
    mol = Chem.RWMol()
    conf = Chem.Conformer(n_real)
    keep_idx = np.where(mask > 0.5)[0]
    for i, idx in enumerate(keep_idx):
        z = atomic_numbers[int(h_argmax[idx])]
        mol.AddAtom(Chem.Atom(int(z)))
        conf.SetAtomPosition(i, (float(x[idx, 0]), float(x[idx, 1]), float(x[idx, 2])))
    mol.AddConformer(conf)

    # Bond perception via RDKit
    try:
        rwmol = mol.GetMol()
        Chem.SanitizeMol(rwmol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_NONE)
        Chem.rdDetermineBonds.DetermineConnectivity(rwmol)
        Chem.SanitizeMol(rwmol)
        return rwmol
    except Exception:
        return None


def decode_batch(
    x: Tensor, h: Tensor, mask: Tensor, atomic_numbers: Sequence[int] = QM9_ATOMIC_NUMBERS,
) -> list[Optional[Chem.Mol]]:
    x_np = x.detach().cpu().numpy()
    h_np = h.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()
    h_argmax = h_np.argmax(axis=-1)
    return [
        decode_to_mol(x_np[b], h_argmax[b], mask_np[b], atomic_numbers)
        for b in range(x_np.shape[0])
    ]
