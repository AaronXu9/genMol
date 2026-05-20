"""In-process RDKit oracle: validity / uniqueness / novelty / QED / SA."""
from __future__ import annotations

from typing import Any, Optional, Sequence

from rdkit import Chem
from rdkit.Chem import QED, Descriptors

from genmol.oracle.base import Oracle, OracleResult


class RDKitOracle(Oracle):
    name = "rdkit"
    version = "1"

    def __init__(self, score_kind: str = "qed"):
        """score_kind: one of 'qed', 'logp', 'mw'. Determines the scalar returned."""
        if score_kind not in {"qed", "logp", "mw"}:
            raise ValueError(f"Bad score_kind: {score_kind}")
        self.score_kind = score_kind

    def __call__(
        self, mols: Sequence[Any], receptor: Optional[Any] = None
    ) -> list[OracleResult]:
        results: list[OracleResult] = []
        for m in mols:
            if m is None:
                results.append(OracleResult(score=float("nan"), success=False, error="None mol"))
                continue
            if isinstance(m, str):
                m = Chem.MolFromSmiles(m)
                if m is None:
                    results.append(OracleResult(score=float("nan"), success=False, error="parse"))
                    continue
            try:
                if self.score_kind == "qed":
                    score = QED.qed(m)
                elif self.score_kind == "logp":
                    score = Descriptors.MolLogP(m)
                else:
                    score = Descriptors.MolWt(m)
                results.append(
                    OracleResult(
                        score=float(score),
                        metadata={"smiles": Chem.MolToSmiles(m)},
                    )
                )
            except Exception as e:
                results.append(
                    OracleResult(score=float("nan"), success=False, error=str(e))
                )
        return results
