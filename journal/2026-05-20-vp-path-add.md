# vp-path-add

**Type:** iteration
**Date:** 2026-05-20
**Code:** Landed in commit `53f8aa6`; default-switched in `7892155`
**Linked experiment(s):** [[2026-05-20-fm-vp-path]] (the run that validated it)

## Change

Added a variance-preserving cos/sin interpolant path (`VPPath`, Albergo 2023 style) as a sibling to the existing `LinearPath` (Lipman 2023) in `genmol/process/flow_paths.py`. Both implement the `InterpolantPath` ABC, both register into a `PATHS` dict keyed by string. Selecting between them is a Hydra config edit:

```yaml
# configs/process/fm_vp.yaml
_target_: genmol.process.flow_matching.FlowMatchingProcess
path: vp        # was 'linear' in the placeholder before this change
```

```
LinearPath:  x_t = (1-t) x_0 + t x_1            v_target = x_1 - x_0   (constant in t)
VPPath:      x_t = cos(πt/2) x_0 + sin(πt/2) x_1  v_target = (π/2)(-sin(πt/2) x_0 + cos(πt/2) x_1)
```

VP is variance-preserving: if `x_0, x_1 ~ N(0, I)` then `x_t ~ N(0, I)` marginally for any t.

## Why

The original M2 Flow Matching run at 200k steps hit only 61.3% validity, vs the EDM run on the same backbone at 90.0%. After ruling out LR-too-high via [[2026-05-20-fm-lr-5e-5-comparison]] (lower LR was worse), the most plausible remaining culprit is the linear interpolant's constant-velocity target. The model has to map a t-varying input to a t-independent output — an information-theoretically harder learning problem than diffusion's t-conditional eps-prediction.

VP's per-t-varying target velocity gives the same kind of t-conditional signal that diffusion eps-prediction provides, which should make optimization cleaner.

## Files touched

- `genmol/process/flow_paths.py:1-80` — added `VPPath` class, `_broadcast_t` helper, registered in `PATHS` dict
- `configs/process/fm_vp.yaml` — re-pointed at the new path (was aliasing linear as a placeholder)
- `tests/unit/test_flow_paths.py` (new) — 6 tests:
  - Linear boundary conditions (t=0 → x_0, t=1 → x_1)
  - Linear velocity is constant
  - VP boundary conditions
  - VP variance preservation (var ≈ 1.0 across 50k samples at t = 0.25, 0.5, 0.75)
  - VP closed-form velocity matches numerical d/dt
  - Both paths in the registry

Later (commit `7892155`), `configs/experiment/{qm9,crossdocked,plinder}_flow.yaml` were updated to use `process: fm_vp` by default.

## Sanity check

VP variance preservation test (50k samples each):

```
t=0.25: empirical Var(x_t) = 0.998   (target 1.0)
t=0.50: empirical Var(x_t) = 1.001
t=0.75: empirical Var(x_t) = 0.999
```

Closed-form velocity vs central-difference at t=0.4 with eps=1e-4: max diff 1.1e-3 (fp32 round-off limit).

## Outcome

Works. The [[2026-05-20-fm-vp-path]] experiment trained FM with VP for 200k steps under the exact same conditions as the original linear-path run, and reached **92.0% validity** — beating EDM (90.0%) by 2pp and the linear-path FM (61.3%) by 30pp. Mean QED also higher (0.422 vs EDM 0.416).

The val-loss-vs-validity inversion confirmed a separate lesson: val loss is not path-invariant. VP's higher loss magnitude (2.87 vs linear's 1.96) is just π/2 scaling of the targets, not a worse model.

Now the default FM path for QM9 / CrossDocked / PLINDER configs.
