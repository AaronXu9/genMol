# ADR 0002: Oracles via subprocess into other conda envs

## Context

Docking engines (Uni-Dock2) and structure prediction models (Boltz-2) have
strict CUDA/Python pins. Installing them into the `genmol` training env would
create irresolvable dependency conflicts:

- Uni-Dock2: CUDA 12.0, Python 3.10, custom CMake.
- Boltz-2: pinned PyTorch + Jax versions.
- `genmol` training: PyTorch 2.4 + PyG.

## Decision

Define `SubprocessOracle` as a base class. Subclasses set `conda_env` and
`worker_script` attributes. Calling the oracle invokes `conda run -n <env>
python <worker>` with JSON over stdin/stdout.

Each oracle has a `_workers/<name>_worker.py` script that runs INSIDE the
target env and contains all the env-specific imports.

## Consequence

- `genmol` env stays clean.
- An oracle crash doesn't kill training (subprocess returncode is handled).
- Worker scripts can be updated independently of `genmol`.

## Cost

- Per-call overhead: ~1 s of subprocess startup. Mitigated by batching
  (Vina is run on ~100 molecules per call) and the SQLite cache.
- Debugging is slightly harder — stderr from the worker comes back as a
  string in `OracleSubprocessError`.
