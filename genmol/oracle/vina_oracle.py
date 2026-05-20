"""Uni-Dock2 / Vina docking oracle — invoked in conda env 'unidock2' via subprocess.

The actual docking command runs in `genmol/oracle/_workers/vina_worker.py`,
which reads a JSON payload from stdin and writes JSON to stdout.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

from genmol.oracle.base import OracleResult, SubprocessOracle


class VinaOracle(SubprocessOracle):
    name = "vina"
    version = "1"
    conda_env = "unidock2"

    def __init__(
        self,
        exhaustiveness: int = 8,
        num_modes: int = 9,
        timeout: int = 600,
    ):
        self.exhaustiveness = exhaustiveness
        self.num_modes = num_modes
        self.timeout = timeout
        self.worker_script = Path(__file__).resolve().parent / "_workers" / "vina_worker.py"

    def __call__(
        self,
        mols: Sequence[Any],          # expected: list of SDF paths or Mol objects
        receptor: Optional[Any] = None,
    ) -> list[OracleResult]:
        if receptor is None:
            return [
                OracleResult(score=float("nan"), success=False, error="vina requires receptor PDB")
                for _ in mols
            ]
        payload = {
            "receptor": str(receptor),
            "mols": [str(m) for m in mols],
            "exhaustiveness": self.exhaustiveness,
            "num_modes": self.num_modes,
        }
        try:
            resp = self._run(payload)
        except Exception as e:
            return [
                OracleResult(score=float("nan"), success=False, error=str(e))
                for _ in mols
            ]
        return [
            OracleResult(score=float(r["score"]), metadata=r.get("metadata", {}),
                         success=r.get("success", True), error=r.get("error"))
            for r in resp["results"]
        ]
