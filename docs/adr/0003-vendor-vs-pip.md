# ADR 0003: Vendor model code rather than `pip install`

## Context

Three external repos contain model code we want:

- **EGNN** (Satorras 2021) — backbone primitives. Not on PyPI; just reference code.
- **DiffSBDD** (Schneuing 2024) — pocket-conditioned dynamics. Not on PyPI;
  full training framework that we don't want to inherit.
- **FlowMol** (Dunn 2024) — flow-matching loss formulation. Not on PyPI;
  another full training framework.

## Decision

For each: `git clone` the upstream, copy ONLY the model architecture files we
need into `third_party/<name>/`, preserve `LICENSE`, and write a `VENDORED.md`
recording the exact commit SHA and what we use.

Wrappers under `genmol/nn/{egnn,pocket_egnn}.py` import from `third_party/`.

## Consequence

- Frozen reproducibility: the version we trained against never drifts.
- We don't inherit upstream training loops, CLIs, or unrelated dependencies.
- Attribution is unambiguous (LICENSE + VENDORED.md visible at the import site).

## Cost

- We are responsible for picking up upstream bug fixes manually.
- Vendoring drift: must check that what we modify locally is tracked in
  `VENDORED.md`. Aim: NO modifications. If a fix is needed, fork upstream and
  pin our fork.

## Reference

`third_party/README.md` is the index. Per-component details in each
`third_party/<name>/VENDORED.md`.
