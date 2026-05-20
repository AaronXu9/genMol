"""Standalone evaluation harness producing reports/<run>/report.{json,md}."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

from rdkit import Chem

from genmol.eval.metrics_2d import rdkit_metrics
from genmol.eval.metrics_3d import bond_length_stats


def evaluate(
    mols: Iterable[Optional[Chem.Mol]],
    output_dir: str | Path,
    train_smiles: Optional[set[str]] = None,
) -> dict:
    mols = list(mols)
    metrics = {}
    metrics.update(rdkit_metrics(mols, train_smiles=train_smiles))
    metrics.update(bond_length_stats(mols))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(metrics, indent=2))
    md = "# Eval report\n\n" + "\n".join(f"- **{k}**: {v}" for k, v in metrics.items())
    (out_dir / "report.md").write_text(md)
    return metrics
