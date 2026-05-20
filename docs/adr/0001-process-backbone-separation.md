# ADR 0001: Strict process / backbone separation

## Context

Many published implementations (EDM, FlowMol) mix the noise schedule, the loss
formulation, and the equivariant network into a single `nn.Module` subclass.
This makes the "EDM vs Flow Matching" swap a code change rather than a config
change.

## Decision

In `genmol`, the `Backbone` (e.g. `EGNNBackbone`) is a pure prediction module
that returns a tensor whose semantics are determined by `prediction_type`. The
`Process` (`EDMProcess`, `FlowMatchingProcess`) owns:

- corruption sampling
- loss formulation
- reverse-step math

`BaseGenLitModule` composes them via HAS-A. The `prediction_type` on the
backbone and on the process must match — a `ValueError` is raised at construction
time otherwise.

## Consequence

- EDM ↔ FM swap is a one-line Hydra config change (`process: edm_polynomial` ↔ `process: fm_linear`).
- Backbones don't need to know what kind of generative process they serve.
- Adding a new process (e.g. variance-preserving FM, score-matching) doesn't
  touch the backbone code.

## Cost

- Slight verbosity: the `Process` ABC has 7 abstract methods.
- The `prediction_type` cross-check is a runtime guard rather than a type-system
  guarantee. Tradeoff: Python doesn't enforce nominal subtype constraints on
  string-tagged behavior anyway.
