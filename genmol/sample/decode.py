"""Decode generator output tensors into RDKit Mol objects.

Steps:
1. argmax over atom-type logits → atomic numbers.
2. Construct an RDKit Mol with atomic numbers + 3D coords.
3. Run xyz-based bond perception (RDKit's `rdDetermineBonds`, properly imported
   as a submodule — `Chem.rdDetermineBonds` does NOT work).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds  # MUST be imported as submodule
from torch import Tensor

from genmol.utils.chem import QM9_ATOMIC_NUMBERS

# RDKit-domain failures we expect during xyz→mol bond perception. Catching
# ONLY these means API-typo bugs (AttributeError, TypeError, ImportError)
# propagate and crash loudly instead of silently returning None — which is
# what hid the rdDetermineBonds import bug for two 200k-step runs.
_RDKIT_DECODE_ERRORS = (ValueError, RuntimeError, Chem.MolSanitizeException)


def decode_to_mol(
    x: np.ndarray,            # (N, 3)
    h_argmax: np.ndarray,     # (N,) atom-type indices
    mask: np.ndarray,         # (N,) 1=real
    atomic_numbers: Sequence[int] = QM9_ATOMIC_NUMBERS,
    charge: int = 0,
    sanitize: bool = True,
) -> Optional[Chem.Mol]:
    """Build an RDKit Mol from (coords, atom-type indices, mask). Returns None
    on failure.

    Args:
        x:              (N, 3) coordinates (Å).
        h_argmax:       (N,) integer atom-type indices into `atomic_numbers`.
        mask:           (N,) float mask; 1.0 = real atom, 0.0 = padding.
        atomic_numbers: sequence indexed by h_argmax → atomic number.
        charge:         total formal charge (default 0 — used by bond perception).
        sanitize:       run a final `Chem.SanitizeMol` so the molecule is
                        ready for SMILES / QED / etc. Set False to keep mols
                        that pass connectivity-perception but fail valence
                        checks (useful for diagnostics).
    """
    n_real = int(mask.sum())
    if n_real < 2:
        return None
    rw = Chem.RWMol()
    conf = Chem.Conformer(n_real)
    keep_idx = np.where(mask > 0.5)[0]
    for i, idx in enumerate(keep_idx):
        z = atomic_numbers[int(h_argmax[idx])]
        rw.AddAtom(Chem.Atom(int(z)))
        conf.SetAtomPosition(
            i, (float(x[idx, 0]), float(x[idx, 1]), float(x[idx, 2]))
        )
    rw.AddConformer(conf)
    mol = rw.GetMol()

    # Bond perception from xyz. Uses RDKit's port of xyz2mol.
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=charge)
    except _RDKIT_DECODE_ERRORS:
        # Fallback: connectivity only (no bond orders) — still useful for
        # validity checks even when bond-order assignment fails.
        try:
            rdDetermineBonds.DetermineConnectivity(mol, charge=charge)
        except _RDKIT_DECODE_ERRORS:
            return None

    if sanitize:
        try:
            Chem.SanitizeMol(mol)
        except _RDKIT_DECODE_ERRORS:
            return None
    return mol


def decode_batch(
    x: Tensor,
    h: Tensor,
    mask: Tensor,
    atomic_numbers: Sequence[int] = QM9_ATOMIC_NUMBERS,
    sanitize: bool = True,
) -> list[Optional[Chem.Mol]]:
    x_np = x.detach().cpu().numpy()
    h_np = h.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()
    h_argmax = h_np.argmax(axis=-1)
    return [
        decode_to_mol(x_np[b], h_argmax[b], mask_np[b], atomic_numbers, sanitize=sanitize)
        for b in range(x_np.shape[0])
    ]
