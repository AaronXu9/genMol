# decoder-bug-fix

**Type:** iteration
**Date:** 2026-05-20
**Code:** Fix landed in `43a6322`; defensive tightening + regression tests in `d66fe1e`
**Linked experiment(s):** [[2026-05-20-fm-lr-5e-5-comparison]], [[2026-05-20-fm-vp-path]] — both could not have been interpreted without this fix

## Change

Two-part fix to a silently-failing 3D→RDKit decoder:

1. `genmol/sample/decode.py`: replaced `Chem.rdDetermineBonds.DetermineConnectivity(...)` (raises `AttributeError` — `rdDetermineBonds` is a submodule, not an attribute) with `from rdkit.Chem import rdDetermineBonds` + `rdDetermineBonds.DetermineBonds(mol, charge=...)` (full bond-order assignment, fall back to `DetermineConnectivity` on failure).
2. Narrowed `except Exception:` to `except (ValueError, RuntimeError, Chem.MolSanitizeException):` in `decode.py` and `metrics_2d.py` so any future API typo crashes loudly instead of being swallowed.
3. Added `RDKitEvalCallback._maybe_warn_zero_validity` — warns if in-training validity stays exactly 0 for N consecutive evals past warmup. The watchdog that would have surfaced the original bug within a few thousand steps.

## Why

The decoder bug returned `None` for every molecule decoded from sampler output, including for known-good inputs like a synthetic methane geometry. The bare `except Exception:` silently caught the `AttributeError` so:

- The in-training `rdkit_eval` callback reported validity=0.0 for the entire 200k-step run of M1 (EDM) and M2 (linear FM).
- The final `wandb-summary.json` for both runs shows `"rdkit/validity": 0` even though both checkpoints were correctly trained.
- The bug was found only after manually loading the EDM checkpoint and discovering that bond-distance histograms looked sane (mean 1.32 Å, etc.) but the decoder still returned all-None.

This was two 2-hour GPU training runs producing apparently-failed models that were actually fine.

## Files touched

- `genmol/sample/decode.py:1-95` — proper submodule import, narrow except, `sanitize` knob, `charge` parameter
- `genmol/eval/metrics_2d.py:1-50` — narrow except
- `genmol/train/callbacks/rdkit_eval.py:1-95` — `zero_validity_warn_after` parameter + counter logic
- `tests/unit/test_decode_known_mol.py` (new) — 5 regression tests: methane round-trip, water round-trip, n_real<2 returns None, decode_batch count, AttributeError propagation (critical: this last test would have caught the original bug)
- `tests/unit/test_rdkit_eval_callback.py` (new) — 3 tests for the watchdog warning logic

## Sanity check

```python
# Before fix:
mol = decode_to_mol(methane_coords, [1,0,0,0,0], np.ones(5))
# Returns: None (incorrect — methane is valid)

# After fix:
mol = decode_to_mol(methane_coords, [1,0,0,0,0], np.ones(5))
# Returns: rdkit Mol, Chem.MolToSmiles(Chem.RemoveHs(mol)) == "C"
```

Re-benchmarked the existing 200k EDM checkpoint after the fix: **90.0% validity / 100% uniqueness** (matches Hoogeboom 2022's ~91%). The trained model was always correct; only the decoder was broken.

## Outcome

Fixed. All 8 new tests pass; full suite now 33/33 green (was 17 at session start before this fix shipped). CI on `.github/workflows/ci.yml` runs all of them on every PR. Most importantly: the AttributeError-must-propagate test makes this exact bug class unable to recur.

The defensive lesson: in Python, `except Exception:` should be a code smell unless paired with logging or re-raise. Narrow the catch list to the domain errors you actually expect.
