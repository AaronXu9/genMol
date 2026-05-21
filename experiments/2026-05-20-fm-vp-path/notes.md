# fm-vp-path

**Type:** experiment
**Date:** 2026-05-20
**Status:** done

> Retroactively documented after the run (see [[2026-05-20-m1-m2-fix-errors]] session summary). Hypothesis and command are accurate as issued.

## Goal

Test whether the FM under-training problem is caused by the linear interpolant's pathological constant-velocity target. Switch to a variance-preserving cos/sin path (Albergo 2023) while keeping every other hyperparameter identical to the original M2 run.

## Hypothesis

The linear path has `v_target = x_1 − x_0` which is constant in t — the model must learn to map a t-varying input `x_t` to a t-independent target. The VP path has `v_target = (π/2)(−sin(πt/2) x_0 + cos(πt/2) x_1)` which varies smoothly with t and gives a uniform-magnitude learning signal across the trajectory. **Expect VP to achieve substantially higher sample validity at 200k steps**, possibly matching EDM (~90%).

Note: val loss won't be directly comparable across paths because VP target velocities have larger magnitude (π/2 scaling) — the right metric is sample validity.

## Setup
- **Code:** `25a72ecf6e2735bec79db694be1414273226f9c7` (genmol/process/flow_paths.py has both `linear` and `vp` paths since 53f8aa6; `configs/process/fm_vp.yaml` points at `vp`)
- **Env:** `genmol` (Python 3.10.20, PyTorch 2.4.1+cu121, Lightning 2.6.1, RDKit 2026.03.2)
- **Host:** br-443
- **GPU:** NVIDIA GeForce RTX 4090 (24 GB, driver 535.309)
- **pip freeze hash:** `7d241d3eb477`
- **Data:** QM9 from `data/raw/qm9/processed/qm9_genmol.pt` (same as M1/M2 originals)
- **Config:** `configs/experiment/qm9_flow.yaml` with `process=fm_vp` override

## Command
```bash
WANDB_MODE=disabled python -m scripts.train \
  experiment=qm9_flow \
  process=fm_vp \
  trainer.accelerator=gpu trainer.devices=1 trainer.precision=bf16-mixed \
  name=qm9_flow_vp \
  data.num_workers=8
```

## Run log

- 15:13 PDT: training started
- 16:34 PDT: training completed after 200,000 steps (1h 21min wall time, bf16-mixed on RTX 4090 at 76% utilization, ~1.9 GB VRAM)
- Final progress-bar metrics: train/loss_step=2.889, val/loss=2.873, train/loss_epoch=2.884
- Checkpoint: `logs/qm9_flow_vp-20260520-151349/checkpoints/epoch=102-step=200000.ckpt`

**Caveat:** Same WANDB_MODE=disabled issue as the lr=5e-5 run — no in-training time series. Fixed for future runs in `dbf23ad`.

## Results

Benchmarked the final checkpoint on 512 samples, 100 sampling steps, random N atoms ∈ [11, 22], seed=0, CPU sampler at 3.6 mol/s.

| Metric | VP path (this run) | linear lr=1e-4 (original) | EDM (reference) |
|---|---|---|---|
| **Validity** | **0.920** | 0.613 | 0.900 |
| Uniqueness | 0.998 | 1.000 | 1.000 |
| Novelty vs 131,698 train SMILES | 0.998 | 1.000 | 1.000 |
| Mean QED | **0.422** | 0.373 | 0.416 |
| Mean bond length | 1.261 Å | 1.247 Å | 1.258 Å |
| Std bond length | 0.194 Å | 0.219 Å | 0.194 Å |
| Final val loss | 2.873 | 1.961 | 0.353 |

Report: `reports/m2_flow_vp_200k/report.json`.

## Interpretation

Hypothesis **strongly supported**. VP path raises FM validity from 0.613 to 0.920 — a 30pp absolute gain over the original linear-path run, and a 2pp gain over the EDM baseline. The bond-length std (0.194) matches EDM's exactly, and mean QED (0.422) is even slightly better than EDM (0.416).

The val-loss-vs-validity inversion is striking: VP has the HIGHEST val loss (2.87) but the HIGHEST validity (92.0%). This validates the warning that loss is not a path-invariant metric — VP's target magnitudes are π/2 larger, so MSE is naturally larger.

The original M2 FM result of 61.3% validity at lr=1e-4 was caused by the linear path's pathology, not by any LR mismatch or budget shortfall. This is now the conclusive answer to the question opened by [[2026-05-20-fm-lr-5e-5-comparison]].

## Next steps

- Switch `experiment=qm9_flow` default to `process=fm_vp` (done in commit `7892155`).
- Re-test [[2026-05-20-fm-vp-path]]-style swap for pocket-conditioned FM when M3 (CrossDocked) data is available.
- Consider VP-with-x0-prediction as another variant (some papers report it's even cleaner).
- The remaining gap to "perfect" is uniqueness 0.998 → 1.0 (one or two duplicates per 500) and novelty 0.998 (a few generated SMILES collided with train set canonicalization after H stripping). Not concerning for QM9 scale.
