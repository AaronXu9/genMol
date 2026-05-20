"""Phase-2 stub: DDPO trainer (Black et al. 2023).

Online policy-gradient fine-tuning of a diffusion model with REINFORCE-with-
baseline on the reverse trajectory. Treats each denoising step as an action.

Requires Phase-1 Sampler to expose:
- stochastic_sample(return_trajectory=True, return_log_prob=True)
- the trajectory must be differentiable (no torch.no_grad in step()).
"""

from genmol.rl.policy import PolicyWrapper
from genmol.rl.reward import RewardFn


def train_ddpo(policy: PolicyWrapper, reward_fn: RewardFn, *args, **kwargs):
    raise NotImplementedError("Phase 2: DDPO trainer. See Black 2023.")
