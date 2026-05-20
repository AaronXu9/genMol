"""Oracle ABC + SubprocessOracle base.

Oracles produce a scalar score per molecule. Two flavors:
- In-process (RDKit, PoseBusters python API) — cheap, no isolation.
- Subprocess — runs in a different conda env to avoid CUDA/Python pin conflicts
  with the training env (e.g. Uni-Dock2 needs CUDA 12.0, Boltz needs its own).

`SubprocessOracle` writes a JSON payload to stdin of a worker script that runs
under `conda run -n <env> python <worker>.py`, and reads JSON results from
stdout. This keeps the `genmol` env clean.
"""
from __future__ import annotations

import json
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence


class OracleError(RuntimeError):
    pass


class OracleSubprocessError(OracleError):
    pass


@dataclass
class OracleResult:
    score: float
    metadata: dict = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


class Oracle(ABC):
    """Produces an `OracleResult` per input molecule."""

    name: str
    version: str

    @abstractmethod
    def __call__(
        self,
        mols: Sequence[Any],            # RDKit Mol list, SDF paths, or SMILES strings
        receptor: Optional[Any] = None,  # PDB path for docking; None for non-pocket oracles
    ) -> list[OracleResult]: ...

    def vectorized_score(self, mols, receptor=None):
        """Convenience: returns a list[float] of scores (NaN for failures)."""
        results = self(mols, receptor=receptor)
        return [r.score if r.success else float("nan") for r in results]


class SubprocessOracle(Oracle):
    """Runs the heavy-lifting in a different conda env via `conda run`.

    Subclasses set:
        conda_env:     str  — e.g. "unidock2", "boltzina_env"
        worker_script: Path — absolute path to a python script that reads JSON
                              from stdin and writes JSON to stdout.
        timeout:       int  — seconds (default 600)
    """

    conda_env: str
    worker_script: Path
    timeout: int = 600

    def _run(self, payload: dict) -> dict:
        cmd = [
            "conda",
            "run",
            "-n",
            self.conda_env,
            "--no-capture-output",
            "python",
            str(self.worker_script),
        ]
        try:
            proc = subprocess.run(
                cmd,
                input=json.dumps(payload).encode(),
                capture_output=True,
                check=False,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise OracleSubprocessError(
                f"{self.name} timed out after {self.timeout}s"
            ) from e

        if proc.returncode != 0:
            raise OracleSubprocessError(
                f"{self.name} (env={self.conda_env}) exited {proc.returncode}: "
                f"{proc.stderr.decode(errors='replace')[:2000]}"
            )

        try:
            return json.loads(proc.stdout.decode())
        except json.JSONDecodeError as e:
            raise OracleSubprocessError(
                f"{self.name} returned non-JSON stdout: "
                f"{proc.stdout.decode(errors='replace')[:500]}"
            ) from e
