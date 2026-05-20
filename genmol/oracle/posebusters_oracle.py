"""PoseBusters-based 3D geometry oracle. M4+."""
from __future__ import annotations

from typing import Any, Optional, Sequence

from genmol.oracle.base import Oracle, OracleResult


class PoseBustersOracle(Oracle):
    name = "posebusters"
    version = "1"

    def __init__(self, **kwargs):
        # Defer the heavy import.
        pass

    def __call__(self, mols: Sequence[Any], receptor: Optional[Any] = None):
        try:
            from posebusters import PoseBusters
        except ImportError as e:
            return [
                OracleResult(score=float("nan"), success=False, error=f"posebusters not installed: {e}")
                for _ in mols
            ]
        buster = PoseBusters(config="mol")
        results: list[OracleResult] = []
        # PoseBusters expects either SDF paths or RDKit Mols; here we route
        # the latter. For batch efficiency a real impl writes a single SDF.
        for m in mols:
            if m is None:
                results.append(OracleResult(score=float("nan"), success=False, error="None mol"))
                continue
            try:
                report = buster.bust(m)
                pass_rate = float(report.iloc[0].astype(bool).mean())
                results.append(OracleResult(score=pass_rate, metadata={"report": report.to_dict()}))
            except Exception as e:
                results.append(OracleResult(score=float("nan"), success=False, error=str(e)))
        return results
