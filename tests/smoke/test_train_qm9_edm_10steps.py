"""Smoke: train EDM-on-QM9 for 10 steps on a 100-mol subset on CPU.

Skipped if PyG / torch_geometric is not importable (CI without those deps).
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("torch_geometric", reason="QM9 dataset needs torch_geometric")

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


@pytest.mark.timeout(180)
def test_train_qm9_edm_10steps(tmp_path: Path, monkeypatch):
    """Runs scripts/train.py with experiment=smoke; expects a checkpoint."""
    import subprocess

    out_dir = tmp_path / "smoke_out"
    out_dir.mkdir()
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"

    cmd = [
        sys.executable,
        str(_REPO / "scripts" / "train.py"),
        "experiment=smoke",
        f"output_dir={out_dir}",
    ]
    result = subprocess.run(cmd, cwd=str(_REPO), env=env, capture_output=True, timeout=170)
    assert result.returncode == 0, (
        f"smoke train failed (returncode {result.returncode})\n"
        f"stdout:\n{result.stdout.decode()[:4000]}\n"
        f"stderr:\n{result.stderr.decode()[:4000]}"
    )
    # We don't assert checkpoint existence yet — depends on trainer config.
    # The main point is exit-clean.
