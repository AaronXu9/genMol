"""Helper for invoking commands in a different conda env via subprocess."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Optional


def run_in_conda_env(
    env_name: str,
    script_path: Path,
    payload: Optional[dict] = None,
    timeout: int = 600,
    capture_output: bool = True,
) -> dict:
    """Run `python <script_path>` inside conda env `env_name`. If `payload`
    is given, it is JSON-written to stdin. Stdout is JSON-decoded and returned.
    """
    cmd = ["conda", "run", "-n", env_name, "--no-capture-output", "python", str(script_path)]
    proc = subprocess.run(
        cmd,
        input=(json.dumps(payload).encode() if payload is not None else None),
        capture_output=capture_output,
        check=True,
        timeout=timeout,
    )
    if capture_output:
        return json.loads(proc.stdout.decode())
    return {}
