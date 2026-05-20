# ADR 0004: Hydra experiment-level composition

## Context

We need to run many configurations: {EDM, FM} × {QM9, CrossDocked, PLINDER} ×
{small backbone, base backbone, ...}. Hardcoding combinations would be
unmaintainable; bare CLI flags would be error-prone.

## Decision

Use Hydra with hierarchical config groups under `configs/`:

- Leaf groups: `model/`, `backbone/`, `process/`, `data/`, `trainer/`,
  `callbacks/`, `sampler/`, `oracle/`, `logger/`.
- A single `experiment/<name>.yaml` composes one entry from each leaf group.
- The top-level `config.yaml` selects an experiment via `experiment=<name>`.

`scripts/train.py` is a single Hydra entrypoint:
```bash
python -m scripts.train experiment=qm9_edm
python -m scripts.train experiment=qm9_flow
python -m scripts.train experiment=smoke
```

## Consequence

- Each experiment file is ~20 lines and self-documents its choices.
- Swap diffusion ↔ FM by changing one line: `process: edm_polynomial` →
  `process: fm_linear`.
- Hyperparameter sweeps via Hydra's `multirun` come for free.

## Cost

- Hydra has a learning curve (interpolation, `_target_`, composition order).
- Type errors surface at instantiation rather than at parse time. Mitigation:
  every component is exercised by a smoke test.
