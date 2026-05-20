"""Phase-2 seam: PreferenceDataset ABC for DPO."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from torch.utils.data import Dataset


class PreferenceDataset(Dataset, ABC):
    """Yields (winner, loser, cond) triples. `winner` and `loser` are dicts
    with keys 'x','h','mask' (same shape as the corresponding model's batch).
    `cond` may be None or a pocket-conditioning dict."""

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> tuple[dict, dict, Optional[dict]]: ...
