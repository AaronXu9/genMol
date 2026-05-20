# Phase 1 Spec

Reflects the locked-in design from `/home/aoxu/.claude/plans/the-purpose-of-this-humming-mitten.md`.

## Goals

| | Goal | Dataset | Verifier |
|---|---|---|---|
| **Phase 1a #1** | EDM-style diffusion on QM9 | QM9 (auto-download) | validity >85%, uniqueness >95% (matches Hoogeboom 2022) |
| **Phase 1a #2** | Flow Matching on QM9 sharing the EDM backbone | QM9 | parity with M1 + EDM↔FM swap is one config line |
| **Phase 1b #1** | Pocket-conditioned EDM on CrossDocked2020 | CrossDocked2020 | validity >80%, plausible Vina scores |
| **Phase 1b #2** | Pocket-conditioned FM + full eval pipeline | CrossDocked2020 | full report.md w/ PoseBusters + Vina + Boltz |

Plus:
- PLINDER preprocessing path scaffolded (PLINDER v2 at `/mnt/katritch_lab2/aoxu/2024-06/v2/`).
- All Phase-2 seams (RL, guidance) wired in but stubbed.

## Critical contracts (tested in `tests/unit/`)

1. **EGNN equivariance.** `pred(R·x) == R·pred(x)`, max error < 1e-6.
2. **Process boundary.** EDM polynomial schedule satisfies `alpha(0)≈1`, `alpha(1)≈0`. FM linear interpolant: `x_t≈x_data` at t=1.
3. **EDM↔FM swap.** `BaseGenLitModule` works identically with `EDMProcess` and `FlowMatchingProcess`; only `prediction_type` differs.
4. **Phase-2 gradient flow.** `Sampler.stochastic_sample(return_log_prob=True)` returns a Tensor with `requires_grad=True`, and `loss.backward()` populates backbone gradients.
5. **Phase-2 guidance no-op.** `Sampler.sample(guidance_fn=None) == Sampler.sample(guidance_fn=zero_guidance, scale=1.0)` (modulo float noise).

## Out of scope for Phase 1

- Actual RL training (DDPO/FlowGRPO/DPO) — stubs only.
- Physics-guided sampling implementations (FF, docking-gradient) — stubs only.
- PoseBusters / Vina / Boltz Phase-1 evaluators are wired in via callbacks but
  full Phase-1 milestone gating only depends on RDKit metrics.
