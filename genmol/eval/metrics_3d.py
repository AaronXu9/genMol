"""3D geometry metrics — PoseBusters pass-rate, bond-length stats. M4+."""
from __future__ import annotations

from typing import Iterable, Optional

from rdkit import Chem


def bond_length_stats(mols: Iterable[Optional[Chem.Mol]]) -> dict:
    """Mean/std of bond lengths across a batch — quick proxy for PoseBusters."""
    import numpy as np

    lens = []
    for m in mols:
        if m is None or m.GetNumConformers() == 0:
            continue
        conf = m.GetConformer()
        for bond in m.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            pa, pb = conf.GetAtomPosition(a), conf.GetAtomPosition(b)
            lens.append(((pa.x - pb.x) ** 2 + (pa.y - pb.y) ** 2 + (pa.z - pb.z) ** 2) ** 0.5)
    if not lens:
        return {"mean_bond_length": 0.0, "std_bond_length": 0.0, "n_bonds": 0}
    arr = np.asarray(lens)
    return {
        "mean_bond_length": float(arr.mean()),
        "std_bond_length": float(arr.std()),
        "n_bonds": int(arr.size),
    }
