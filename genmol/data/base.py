"""Dataclasses and ABCs shared across all dataset modules.

`ProcessBatch` lives here (not in `process/`) because it is the data structure
that flows between Data → Backbone → Process; making it a `data` citizen is
cleaner than coupling the process module to data shapes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from torch import Tensor
from torch.utils.data import Dataset


@dataclass
class ProcessBatch:
    """A batched molecular state for the generative process.

    Attributes:
        x:    (B, N, 3) atom coordinates. COM-zeroed per-sample by convention.
        h:    (B, N, F) atom features (typically one-hot atom type).
        mask: (B, N) 1 = real atom, 0 = padding.
        cond: Optional conditioning dict. For pocket-conditioned (SBDD) models:
              {"pocket_x": (B, M, 3), "pocket_h": (B, M, Fp), "pocket_mask": (B, M)}.
              None for unconditional.
        aux:  Process-specific auxiliary tensors (e.g. noise sample, target velocity).
              Populated by `Process.corrupt`, consumed by `Process.loss`.
    """

    x: Tensor
    h: Tensor
    mask: Tensor
    cond: Optional[dict] = None
    aux: dict = field(default_factory=dict)

    def to(self, device) -> "ProcessBatch":
        return ProcessBatch(
            x=self.x.to(device),
            h=self.h.to(device),
            mask=self.mask.to(device),
            cond={k: v.to(device) for k, v in self.cond.items()} if self.cond else None,
            aux={k: v.to(device) if hasattr(v, "to") else v for k, v in self.aux.items()},
        )

    @property
    def batch_size(self) -> int:
        return int(self.x.shape[0])

    @property
    def device(self):
        return self.x.device


class MolDataset(Dataset, ABC):
    """Unconditional 3D molecular dataset (e.g. QM9, GEOM-Drugs)."""

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> dict:
        """Returns a dict with keys: 'x' (N,3), 'h' (N,F), 'mask' (N,)."""


class PocketLigandDataset(Dataset, ABC):
    """Pocket-conditioned dataset (e.g. CrossDocked, PLINDER)."""

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> dict:
        """Returns a dict with ligand keys 'x','h','mask' and pocket keys
        'pocket_x','pocket_h','pocket_mask'."""
