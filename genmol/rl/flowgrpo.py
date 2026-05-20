"""Phase-2 stub: FlowGRPO trainer (Liu 2025).

GRPO (group-relative policy optimization, from DeepSeek-Math) adapted to
flow-matching models via:
1. ODE → SDE conversion for stochastic exploration while preserving marginals.
2. Denoising reduction: few SDE steps in training, full ODE schedule at inference.

Requires Phase-1 Sampler to expose:
- stochastic_sample(return_trajectory=True, return_log_prob=True)
- the log_prob must have requires_grad=True for backprop through the policy.
"""

from genmol.rl.policy import PolicyWrapper
from genmol.rl.reward import RewardFn


def train_flowgrpo(policy: PolicyWrapper, reward_fn: RewardFn, *args, **kwargs):
    raise NotImplementedError("Phase 2: FlowGRPO trainer. See Liu 2025.")
