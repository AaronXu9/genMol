"""Phase-2 stub: DPO trainer (Rafailov 2023, adapted to diffusion/FM).

Offline preference fine-tuning. Given pairs (winner, loser, cond), train the
generator to maximize the log-prob ratio between winner/loser relative to a
frozen reference model. Diffusion-DPO (Wallace 2023) is the closest analog.

Requires:
- A frozen reference model (genmol.rl.ref_model.freeze).
- A PreferenceDataset of (winner, loser, cond) tuples.
"""

from genmol.rl.preference_dataset import PreferenceDataset
from genmol.rl.policy import PolicyWrapper


def train_dpo(policy: PolicyWrapper, dataset: PreferenceDataset, *args, **kwargs):
    raise NotImplementedError("Phase 2: DPO trainer. See Wallace 2023.")
