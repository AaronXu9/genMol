"""Boltz-2 affinity oracle — invoked in conda env 'boltzina_env' via subprocess."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

from genmol.oracle.base import OracleResult, SubprocessOracle


class BoltzOracle(SubprocessOracle):
    name = "boltz"
    version = "2"
    conda_env = "boltzina_env"

    def __init__(self, timeout: int = 3600):
        self.timeout = timeout
        self.worker_script = Path(__file__).resolve().parent / "_workers" / "boltz_worker.py"

    def __call__(
        self,
        mols: Sequence[Any],
        receptor: Optional[Any] = None,
    ) -> list[OracleResult]:
        if receptor is None:
            return [
                OracleResult(score=float("nan"), success=False, error="boltz requires receptor")
                for _ in mols
            ]
        payload = {"receptor": str(receptor), "mols": [str(m) for m in mols]}
        try:
            resp = self._run(payload)
        except Exception as e:
            return [
                OracleResult(score=float("nan"), success=False, error=str(e))
                for _ in mols
            ]
        return [
            OracleResult(
                score=float(r["affinity"]),
                metadata={"iptm": r.get("iptm"), "plddt": r.get("plddt")},
                success=r.get("success", True),
                error=r.get("error"),
            )
            for r in resp["results"]
        ]
