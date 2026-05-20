# genMol

Generative molecular modeling: a shared scaffold for diffusion-based (EDM-style) and flow-matching-based (FlowMol-style) molecular generators, with first-class seams for RL fine-tuning (DDPO/FlowGRPO), DPO, and physics-guided sampling.

## Scope

- **Phase 1a**: Unconditional 3D molecule generation on QM9.
- **Phase 1b**: Pocket-conditioned SBDD on CrossDocked2020 (with PLINDER preprocessing scaffolded).
- **Phase 2**: RL fine-tuning + physics-guided sampling, hooked into the Phase-1 generator without refactor.

## Quick start

```bash
# 1. Create conda env
make env
conda activate genmol
make install

# 2. Smoke test (CPU, <2 min)
make smoke

# 3. Train Phase 1a EDM on QM9
make prep-qm9
make train-qm9-edm

# 4. Train Phase 1a Flow Matching (same backbone, just process swap)
make train-qm9-fm
```

## Milestones

| | Goal | Verifier |
|---|---|---|
| **M0** | Skeleton + smoke test | `make smoke` |
| **M1** | QM9 + EDM → first checkpoint | validity >85%, uniqueness >95% on val (matches Hoogeboom 2022) |
| **M2** | QM9 + Flow Matching, same backbone | parity with M1; EDM↔FM is a one-line config swap |
| **M3** | CrossDocked + pocket-conditioned EDM | pocket-conditioned validity >80%, plausible Vina scores |
| **M4** | CrossDocked + Flow Matching + full eval | full report.md with PoseBusters/Vina/Boltz |
| **M5** | Phase-2 seam sanity check | `pytest tests/unit/test_phase2_seams.py` |

## Architecture (one-paragraph)

The training stack composes three independent pieces: a `Backbone` (E(n)-equivariant network — predicts a tensor whose semantics depend on `prediction_type`), a `Process` (EDM diffusion or Flow Matching — owns corruption/loss/reverse-step), and a `Sampler` (inference with first-class `guidance_fn` and `return_log_prob` hooks for Phase-2 RL). Lightning glues them via `LitModule`s; Hydra configures them via `configs/experiment/*.yaml`. See `docs/ARCHITECTURE.md`.

## Layout

See the plan at `docs/PHASE1_SPEC.md` (mirrors `/home/aoxu/.claude/plans/the-purpose-of-this-humming-mitten.md`).

## Vendored dependencies

See `third_party/README.md`. EGNN is vendored from `/home/aoxu/projects/egnn/` (MIT). DiffSBDD and FlowMol references are fetched on demand via `tools/`.

## License

MIT. Vendored components retain their original licenses under `third_party/`.
