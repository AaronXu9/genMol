# Oracles

All oracles share a single interface (`genmol/oracle/base.py:Oracle`):

```python
oracle(mols, receptor=None) -> list[OracleResult]
```

Where each `OracleResult` has `.score`, `.metadata`, `.success`, `.error`.

## In-process oracles

### RDKitOracle
- 2D metrics: validity, QED, logP, molecular weight.
- ~1 ms/molecule (CPU).
- Use: training-time callback (`rdkit_eval`), default scorer in RL reward composition.

### PoseBustersOracle
- 3D-geometry plausibility checks (bond lengths, angles, stereo).
- ~100 ms – 1 s/molecule (CPU).
- Use: periodic eval callback (`posebusters_eval`).

## Subprocess oracles

These run in a different conda env via `conda run`. The genmol env never imports them.

### VinaOracle (env: `unidock2`)
- Wraps Uni-Dock2 GPU docking.
- ~1–10 s/molecule (GPU).
- Use: Phase-1b sparse eval, Phase-2 reward.
- Worker: `genmol/oracle/_workers/vina_worker.py`.

Invocation pattern:
```
conda run -n unidock2 --no-capture-output python genmol/oracle/_workers/vina_worker.py
# JSON in on stdin → JSON out on stdout
```

### BoltzOracle (env: `boltzina_env`)
- Wraps Boltz-2 affinity prediction.
- ~1–5 min/complex (GPU).
- Use: nightly top-K rescoring, NOT training-time.
- Worker: `genmol/oracle/_workers/boltz_worker.py`.

## Why subprocess isolation?

- Uni-Dock2 requires CUDA 12.0 + Python 3.10 + custom CMake build.
- Boltz pins specific PyTorch / Jax versions for the structure prediction model.
- Forcing them into the same env as our PyTorch 2.4 training stack would cause
  irresolvable conflicts.
- Subprocess isolation is also a clean robustness boundary: an oracle crash
  doesn't kill training.

## Caching

`genmol/oracle/cache.py:OracleCache` is a SQLite cache keyed by
`(smiles_canonical, receptor_sha1, oracle_name, oracle_version)`. Re-running
an oracle on the same molecule is a no-op DB lookup.

## Phase-2: composite reward

```python
from genmol.rl.reward import WeightedOracleReward
from genmol.oracle.rdkit_oracle import RDKitOracle
from genmol.oracle.vina_oracle import VinaOracle

reward = WeightedOracleReward(
    oracles=[RDKitOracle(score_kind='qed'), VinaOracle()],
    weights=[0.3, 0.7],
)
```

(Implementation gated to Phase 2 — see `genmol/rl/reward.py`.)
