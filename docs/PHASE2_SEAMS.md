# Phase-2 Seams

Three classes of Phase-2 hooks are committed in Phase 1 to prevent a refactor:

## 1. Gradient-bearing log-prob for online RL

**Where**: `genmol/sample/sampler.py:Sampler.stochastic_sample(return_log_prob=True)` + `genmol/sample/logprob.py`.

**Contract**: returns a `(B,)` Tensor with `requires_grad=True`. Calling `loss = -log_prob.mean(); loss.backward()` populates gradients on every trainable parameter of the backbone used in the sampling rollout.

**Phase-2 use**: DDPO (Black 2023) and FlowGRPO (Liu 2025) both compute `policy_gradient = log_prob * (reward - baseline)`. This works only if `log_prob` carries gradient back to the policy parameters.

**Anti-pattern to NEVER introduce**: `with torch.no_grad():` anywhere inside `logprob.py` or inside the sampling loop when `return_log_prob=True`. The grad-flow test (`tests/unit/test_sampler_logprob.py`) is the canary.

## 2. Guidance-fn seam for physics-guided sampling

**Where**: `genmol/sample/sampler.py:Sampler.sample(guidance_fn=...)` + `genmol/sample/guidance.py:zero_guidance, compose_guidance`.

**Contract**: `guidance_fn(x_t, t, cond) → grad_term` of same shape as `x_t`. Sampler adds `guidance_scale * grad_term` to the score/velocity at each reverse step. Passing `guidance_fn=None` (default) is identical to passing `zero_guidance` with any scale.

**Phase-2 use**: Replace `None` with `FFEnergyGuidance(kind='uff')` or a learned docking-gradient surrogate (`DockingGradientSurrogate`). No retraining required; pure inference-time mechanism.

**Anti-pattern**: any `if guidance_fn is None: <special path>` divergence that means the trivial `zero_guidance` produces different outputs from `None`. The seam test (`tests/unit/test_sampler_guidance.py`) is the canary.

## 3. Subprocess oracle seam for reward signals

**Where**: `genmol/oracle/base.py:SubprocessOracle` + `genmol/oracle/_workers/*.py`.

**Contract**: a `SubprocessOracle` subclass sets `conda_env` and `worker_script` attributes. Calling `oracle(mols, receptor=...)` runs `conda run -n <env> python <worker.py>` with a JSON payload on stdin and parses JSON results from stdout.

**Phase-2 use**: `RewardFn` aggregates `VinaOracle`, `BoltzOracle`, `RDKitOracle` into a per-sample scalar reward. Each oracle is its own conda env — never installed into `genmol`.

**Anti-pattern**: importing `unidock2` or `boltz` into the `genmol` env (will cause CUDA/Python version conflicts).

## Reference-model loading

**Where**: `genmol/rl/ref_model.py`.

**Contract**: load a Phase-1 Lightning checkpoint into a frozen `nn.Module` (`requires_grad_(False)` on every parameter, `.eval()` mode). Used as the KL reference in PPO/GRPO or as the "loser model" baseline in DPO.

## Files containing only stubs (raise NotImplementedError)

- `genmol/rl/ddpo.py` — DDPO trainer
- `genmol/rl/flowgrpo.py` — FlowGRPO trainer
- `genmol/rl/dpo.py` — DPO trainer
- `genmol/guidance/ff_energy.py` — Force-field gradient guidance
- `genmol/guidance/docking_surrogate.py` — Learned docking gradient
- `genmol/nn/pocket_egnn.py` — M3 pocket backbone
- `genmol/data/crossdocked.py` — M3 datamodule
- `genmol/data/plinder.py` — M3+ datamodule

Each stub points at the file it'll borrow ideas from (DiffSBDD dynamics, Wallace's Diffusion-DPO, Liu's FlowGRPO paper).
