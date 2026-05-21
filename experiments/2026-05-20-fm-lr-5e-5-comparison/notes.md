# fm-lr-5e-5-comparison

**Type:** experiment
**Date:** 2026-05-20
**Status:** done

> Retroactively documented after the run (see [[2026-05-20-m1-m2-fix-errors]] session summary). Hypothesis and command are accurate as issued; results are from the actual completed run.

## Goal

Test whether the original Flow Matching run's under-training (61.3% validity vs EDM's 90.0%) was caused by the learning rate being too high. Retrain with lr=5e-5 (half) holding everything else constant.

## Hypothesis

If LR was too high, lower LR will produce a smoother loss curve, lower final val loss, and higher sample validity at 200k steps. If the under-training is caused by something else (e.g., path geometry), lower LR will produce slower convergence and worse final validity.

## Setup
- **Code:** `25a72ecf6e2735bec79db694be1414273226f9c7` on branch `main` (clean)
- **Env:** `genmol` (Python 3.10.20, PyTorch 2.4.1+cu121, Lightning 2.6.1, RDKit 2026.03.2)
- **Host:** br-443
- **GPU:** NVIDIA GeForce RTX 4090 (24 GB, driver 535.309)
- **pip freeze hash:** `7d241d3eb477`
- **Data:** QM9 from `data/raw/qm9/processed/qm9_genmol.pt` — 131,970 mols (1,915 skipped from gdb9.sdf)
- **Config:** `configs/experiment/qm9_flow.yaml` (linear path) with `model.lr=5e-5` override

## Command
```bash
WANDB_MODE=disabled python -m scripts.train \
  experiment=qm9_flow \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.precision=bf16-mixed \
  model.lr=5e-5 \
  name=qm9_flow_lr5e5 \
  data.num_workers=8
```

## Run log

- 13:53 PDT: training started
- 15:12 PDT: training completed after 200,000 steps (1h 18min wall time, faster than the original fp32 run because of bf16-mixed)
- Final progress-bar metrics: train/loss_step=2.093, val/loss=2.030, train/loss_epoch=2.039
- Checkpoint: `logs/qm9_flow_lr5e5-20260520-135304/checkpoints/epoch=102-step=200000.ckpt`

**Caveat:** `WANDB_MODE=disabled` + no CSV logger meant in-training validity time series was not persisted. Only the final progress-bar values and the saved checkpoints survived. This is the trigger that motivated commit `dbf23ad` (add CSV logger fallback so the same mistake can't happen again).

## Results

Benchmarked the final checkpoint with `scripts/run_benchmark.py` on 512 samples, 100 sampling steps, random N atoms ∈ [11, 22], seed=0, on CPU.

| Metric | lr=5e-5 (this run) | lr=1e-4 (original baseline) |
|---|---|---|
| Validity | **0.404** | 0.613 |
| Uniqueness | 1.000 | 1.000 |
| Novelty vs 131,698 train SMILES | 1.000 | 1.000 |
| Mean QED | 0.382 | 0.373 |
| Mean bond length | 1.267 Å | 1.247 Å |
| Std bond length | 0.243 Å | 0.219 Å |
| Final val loss | 2.030 | 1.961 |

Report: `reports/m2_flow_lr5e5_200k/report.{json,md}`.

## Interpretation

Hypothesis **refuted**. Lower LR produced *lower* validity (40.4% vs 61.3%) and *higher* val loss (2.030 vs 1.961). The 200k-step budget at lr=5e-5 is insufficient — the model is still mid-convergence when the cosine schedule decays it to near zero. This rules out "LR was too high" as the explanation for the original FM under-training and points to either path geometry (linear interpolant's constant-velocity target) or insufficient steps as the real cause.

The "predict-zero baseline" loss for the linear path is ~3.13 (computed in [[2026-05-20-m1-m2-fix-errors]]), so 2.03 represents only a 35% improvement — well short of EDM's 82%.

## Next steps

- Try VP cos/sin path with the original lr=1e-4 — variance-preserving and t-conditional learning signal might decouple path quality from training budget. → [[2026-05-20-fm-vp-path]]
- If VP also under-performs, the next test would be constant LR (no cosine) for 500k+ steps.
