"""SubprocessOracle wiring sanity (does NOT require unidock2/boltzina_env)."""
from __future__ import annotations

from pathlib import Path

from genmol.oracle.base import SubprocessOracle


def test_subprocess_oracle_has_required_attrs():
    # Subclassing pattern: must set conda_env and worker_script.
    class _Mock(SubprocessOracle):
        name = "mock"
        version = "1"
        conda_env = "nonexistent_env_for_test"
        worker_script = Path("/tmp/nonexistent_worker.py")

        def __call__(self, mols, receptor=None):
            return []

    mock = _Mock()
    assert mock.conda_env == "nonexistent_env_for_test"
    assert mock.worker_script.name == "nonexistent_worker.py"
