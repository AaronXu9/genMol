# ADR 0005: Phase-2 seams committed in Phase 1

## Context

The user's roadmap is:
- **Phase 1**: Train diffusion + flow-matching baselines.
- **Phase 2**: Improve via RL (DDPO/FlowGRPO), DPO, and physics-guided sampling.

A naive Phase 1 implementation would force a refactor in Phase 2:
- Sampler with no `guidance_fn` parameter → physics guidance can't be added
  without changing every caller.
- `log_prob` computed under `torch.no_grad()` → DDPO/FlowGRPO silently broken.
- Oracles installed into the training env → adding Vina/Boltz becomes a
  dependency-resolution problem.

## Decision

Commit three Phase-2 seams in Phase 1:

1. `Sampler.sample(guidance_fn=None)` and `Sampler.stochastic_sample(return_log_prob=False)` accept Phase-2 hooks from day one.
2. `genmol/sample/logprob.py` MUST NOT use `torch.no_grad()` — enforced by
   `tests/unit/test_sampler_logprob.py`.
3. Oracles isolated in subprocesses (see ADR 0002).

Stubs for `PolicyWrapper`, `RewardFn`, `PreferenceDataset`, `FFEnergyGuidance`,
`DockingGradientSurrogate` live in `genmol/rl/` and `genmol/guidance/`. They
raise `NotImplementedError` but define the interface.

## Consequence

- Adding Phase 2 is purely additive — no refactoring of Phase-1 code.
- The seams are continuously exercised (the `guidance_fn=None` path is taken
  every time a sample is generated).
- The tests catch regressions: if someone wraps `logprob.py` in `no_grad`, the
  gradient-flow test fails.

## Cost

- Slight conceptual overhead in Phase 1: parameters like `guidance_fn` and
  `return_log_prob` need to be threaded through.
- The Phase-2 stubs take up file-system space but no runtime cost.
