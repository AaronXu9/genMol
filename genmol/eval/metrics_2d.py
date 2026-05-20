"""RDKit-based 2D metrics: validity / uniqueness / novelty / QED / SA."""
from __future__ import annotations

from typing import Iterable, Optional

from rdkit import Chem
from rdkit.Chem import QED


def rdkit_metrics(mols: Iterable[Optional[Chem.Mol]], train_smiles: set[str] | None = None) -> dict:
    """Compute headline 2D metrics for a batch of (possibly None) RDKit Mols.

    Returns dict with: total, valid, validity, uniqueness, novelty, mean_qed.
    """
    mols = list(mols)
    total = len(mols)
    valid_smiles: list[str] = []
    qed_vals: list[float] = []
    for m in mols:
        if m is None:
            continue
        try:
            s = Chem.MolToSmiles(m)
            if s is None or s == "":
                continue
            valid_smiles.append(s)
            try:
                qed_vals.append(QED.qed(m))
            except Exception:
                pass
        except Exception:
            continue

    n_valid = len(valid_smiles)
    n_unique = len(set(valid_smiles))
    n_novel = (
        len(set(valid_smiles) - train_smiles) if train_smiles is not None else n_unique
    )

    return {
        "total": float(total),
        "valid": float(n_valid),
        "validity": (n_valid / total) if total else 0.0,
        "uniqueness": (n_unique / n_valid) if n_valid else 0.0,
        "novelty": (n_novel / n_valid) if n_valid else 0.0,
        "mean_qed": (sum(qed_vals) / len(qed_vals)) if qed_vals else 0.0,
    }
