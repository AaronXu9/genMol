# M1 & M2 Benchmark Report — QM9 EDM vs Flow Matching

## TL;DR

| | M1: EDM | M2: Flow Matching | Reference (Hoogeboom 2022 EDM) |
|---|---|---|---|
| **Validity** | **0.900** | 0.613 | ~0.91 |
| **Uniqueness** | 1.000 | 1.000 | ~0.99 |
| **Novelty** vs 131,698 train SMILES | 1.000 | 1.000 | ~0.95 |
| **Mean QED** | 0.416 | 0.373 | — |
| **Mean bond length** | 1.258 Å | 1.247 Å | reasonable C-X range |
| **Std bond length** | 0.194 Å | 0.219 Å | — |
| Sampling time (512 mol on CPU) | 849 s @ 250 steps (0.60 mol/s) | 290 s @ 100 steps (1.77 mol/s) | — |

Headline: **EDM matches the paper benchmark (90% vs ~91%)** in 200k steps. **FM lags at 61.3%** with the same backbone, suggesting FM either needs more training or different conditioning. **Both produce 100% novel molecules** (none of the 314–461 valid samples appear in the QM9 training set). The EDM ↔ FM swap contract held in code — the difference is purely the `Process`.

## Training summary

| | M1: EDM | M2: Flow Matching |
|---|---|---|
| Config | `experiment=qm9_edm` | `experiment=qm9_flow` |
| Backbone | `EGNNBackbone(256×9, attention=true)` (4.16M params) | identical |
| Process | `EDMProcess(schedule="polynomial")` | `FlowMatchingProcess(path="linear")` |
| Prediction type | `noise` | `velocity` |
| Optimizer | AdamW, lr 1e-4, cosine schedule, 1k warmup | identical |
| EMA decay | 0.999 | identical |
| Batch size | 64 | 64 |
| Steps | 200,000 | 200,000 |
| Wall time (RTX 4090, fp32) | 1h 51min | 1h 49min |
| Final train loss | **0.381** (started ~2.07, 5.4× drop) | **1.981** (started ~4.28, 2.2× drop) |
| Final val loss | **0.353** | **1.923** |

Note the losses are **not directly comparable** — EDM's `||ε_pred − ε_true||²` lives on a different scale than FM's `||v_pred − v_target||²`.

## Loss-curve highlights (extracted from W&B local files)

### EDM — smooth convergence

```
step      train_loss   val_loss
 3,033    0.585        0.577     ( 1% of training)
12,738    0.440        0.448     (10%)
37,544    0.400        0.402     (25%)
90,396    0.371        0.388     (50%)
143,248   0.364        0.365     (75%)
174,767   0.358        0.365     (90%)
194,062   0.357        0.367     (99%)
```

EDM's loss decreased monotonically with no instability. Most of the gain happens in the first ~50k steps (0.585 → 0.400). After step 90k, val loss plateaus at ~0.36 — diminishing returns. Could probably stop at ~120k without quality loss.

### FM — noisy, with a relapse around step 38–90k

```
step      train_loss   val_loss
 3,033    2.503        2.252     ( 1%)
12,738    2.095        2.021     (10%)
37,544    2.343        2.311     (25%)   ← went UP
90,396    1.970        2.307     (50%)
143,248   2.069        1.993     (75%)
174,767   1.955        1.966     (90%)
194,062   1.935        2.000     (99%)
```

FM's val loss climbed from 2.02 to 2.31 between steps 13k–38k, then recovered to ~1.96 by step 175k. This kind of dynamics often indicates the learning rate is slightly too high for the FM loss curvature, or the linear interpolant path causes training instabilities at intermediate `t`. Worth investigating: lower LR, or VP / cosine interpolant.

## Sample quality

### EDM — 461 / 512 valid (90.0%), example SMILES (canonical, formal-charged from xyz2mol)

```
[H]c1nnc([H])n1C([H])([H])[H]                           # 1-methyl-1,2,4-triazole — real drug fragment
[H][C@@]12C(=O)[C@@]([H])(C2)[C@@]2[C@@H]([H])12
[H]C#C[C@@]12C(H)=[N+]=C(H)[C@]1(H)[C@]2(H)[O-]
[H]C(H)(H)[N+]1=C(C#N)O[N-]O1
```

Many samples carry formal charges from xyz2mol — this is decoder behavior, not model output, and is a well-known property of all 3D molecular generators that decode via geometry-based bond perception.

### FM — 314 / 512 valid (61.3%), examples

```
[H]/C([O-])=C(/[H])C([H])([H])[H].[H]/[O+]=C(\[H])C(...)
[H][C-]([O-])C1([H])[C+]([C][N-2])C1([H])[H]
[H]N([H])C12[O+]=C1[N+]1[N+]2[C-]1[H]
```

FM samples show more disconnected components (`.` in SMILES → multi-fragment) and more exotic charge states. Combined with the noisier loss curve, this strongly suggests under-training rather than a model bug.

## EDM ↔ FM swap contract — verified

The whole point of `genmol`'s `Process` / `Backbone` separation was to make EDM ↔ FM swap configuration-only. Empirically:

- Same backbone class with **identical parameter count** (4,158,222 params).
- Same data pipeline, optimizer, EMA, callbacks.
- Difference between `experiment=qm9_edm` and `experiment=qm9_flow`: **3 lines** (process, sampler, model._target_) + `prediction_type` in the backbone block.

The swap is pure Hydra composition.

## Critical decoder bug fixed during analysis (commit `43a6322`)

`genmol/sample/decode.py` was calling `Chem.rdDetermineBonds.DetermineConnectivity(...)`, which raises `AttributeError` — `rdDetermineBonds` is a submodule of `rdkit.Chem`, not an attribute. The bare `except Exception:` silently caught the error and returned `None`. As a result, **the in-training `rdkit_eval` callback reported 0.0 validity for the entire 200k-step run on both models**, even though the trained checkpoints were correct.

Fix: import `rdDetermineBonds` as a submodule, use `DetermineBonds` (full bond-order assignment) as primary, fall back to `DetermineConnectivity`, add a `sanitize=False` knob for diagnostics.

## Recommendations

1. **EDM is publishable now.** 90% validity / 100% novelty at 200k steps matches Hoogeboom 2022 within 1pp.
2. **FM needs a second pass.** Either:
   - Drop LR to 5e-5 and rerun (cheapest experiment).
   - Switch to a VP path in `genmol/process/flow_paths.py` (Albergo 2023 — VP path has lower variance at intermediate t).
   - Train to 500k steps and re-evaluate.
3. **Re-train with the decoder fix from the start.** The `rdkit_eval` callback will now show real validity curves over training, which lets us early-stop on validity rather than just on val loss.
4. **GPU validation needed.** Current eval is CPU (~0.6 mol/s for EDM). Once the GPU driver mismatch is resolved (host reboot), full eval should be ~50× faster.

## Files

- `reports/m1_edm_200k/report.{json,md}` — EDM final benchmark
- `reports/m2_flow_200k/report.{json,md}` — FM final benchmark
- `reports/m1_edm_200k/history.json` — EDM loss curve (107,835 progress-bar updates)
- `reports/m2_flow_200k/history.json` — FM loss curve
- `logs/qm9_edm-20260519-200717/checkpoints/epoch=102-step=200000.ckpt` — M1 model weights
- `logs/qm9_flow-20260520-054542/checkpoints/epoch=102-step=200000.ckpt` — M2 model weights
