# Architecture

## One-paragraph summary

The training stack composes three independent pieces: a `Backbone` (an E(n)-equivariant network — predicts a tensor whose semantics depend on `prediction_type`), a `Process` (EDM diffusion or Flow Matching — owns corruption, loss, and reverse-step math), and a `Sampler` (inference with first-class `guidance_fn` and `return_log_prob` hooks for Phase-2 RL). `BaseGenLitModule` glues them via composition; Hydra configures them via `configs/experiment/*.yaml`. The contract that lets EDM ↔ FM be a one-line config swap is enforced by the `Process` ABC: the backbone never knows whether it's serving diffusion or flow matching.

```
                 ┌─────────────────────────┐
                 │   BaseGenLitModule      │  ← Lightning glue
                 │  - HAS A Process        │
                 │  - HAS A Backbone       │
                 └────────┬────────────────┘
                          │
       ┌──────────────────┴──────────────────┐
       ▼                                      ▼
┌──────────────────┐                  ┌──────────────────┐
│   Process (ABC)  │                  │  Backbone (ABC)  │
│  - corrupt()     │  --backbone()-->  │  - forward()    │
│  - loss()        │  <-- pred -----  │   predict tensor │
│  - step()        │                  │   (no loss math) │
│  - init_noise()  │                  └──────────────────┘
└──────────────────┘
   │
   ├── EDMProcess (diffusion, prediction_type='noise')
   └── FlowMatchingProcess (FM, prediction_type='velocity')

                          │
                          ▼
                 ┌──────────────────┐
                 │  Sampler (ABC)   │  inference layer
                 │  - sample(...)   │  ← guidance_fn (Phase-2 physics)
                 │  - stochastic_   │  ← return_log_prob (Phase-2 RL)
                 │     sample(...)  │
                 └──────────────────┘
```

## Phase-2 seams (critical)

Three seams baked into Phase 1 from day one:

1. **`Sampler.sample(guidance_fn=...)`** — Phase-2 physics-guided sampling adds a
   gradient term to the score/velocity at each reverse step. Default `None` is
   plumbed through; `zero_guidance` is tested against `None` for identity.

2. **`Sampler.stochastic_sample(return_log_prob=True)`** — Phase-2 RL
   (DDPO/FlowGRPO) requires per-sample log-prob with `requires_grad=True`.
   This is asserted by `tests/unit/test_sampler_logprob.py`. **No `torch.no_grad()`
   may be added inside `genmol/sample/logprob.py`** — doing so silently breaks
   FlowGRPO without failing any test.

3. **`Oracle` + `SubprocessOracle`** — Phase-2 reward signals come from external
   conda envs (`unidock2`, `boltzina_env`). The `SubprocessOracle.run` plumbing
   is in Phase-1 from day one. Phase-1 oracles (RDKit) use the same interface,
   so swapping to Vina/Boltz in Phase 2 is one Hydra-config line.

See `docs/PHASE2_SEAMS.md` for the full contract spec.

## Files of note

| File | Role |
|---|---|
| `genmol/process/base.py` | `Process` ABC |
| `genmol/nn/backbone.py` | `Backbone` ABC |
| `genmol/sample/sampler.py` | `Sampler` ABC + `GuidanceFn` |
| `genmol/oracle/base.py` | `Oracle` + `SubprocessOracle` |
| `genmol/rl/policy.py` | `PolicyWrapper` ABC |
| `genmol/rl/reward.py` | `RewardFn` ABC |
| `genmol/sample/logprob.py` | Gradient-flow contract (no_grad-forbidden) |
| `configs/experiment/*.yaml` | One YAML = one experiment |

## Invariants

- **No process logic inside backbones.** A backbone returns a tensor; the
  Process owns the loss.
- **One backbone, multiple processes.** EDM and FM share the EGNN backbone. The
  only difference is `prediction_type` ('noise' vs 'velocity').
- **No hardcoded paths.** Everything goes through Hydra config.
- **Oracle isolation via subprocess.** Never install Uni-Dock2 or Boltz into
  the `genmol` env — always invoke via `conda run -n <env>`.
