"""QM9 LightningDataModule (custom, RDKit-direct loader).

We do NOT use `torch_geometric.datasets.QM9` because its `process()` step
crashes on newer RDKit (`None` from SDMolSupplier → AttributeError). Instead
we read `gdb9.sdf` directly, filter None mols, and cache the result as a
single `.pt` blob.

Auto-downloads `gdb9.sdf` from the FigShare mirror on first call (PyG-compatible).
"""
from __future__ import annotations

import io
import os
import tarfile
import urllib.request
from pathlib import Path
from typing import Optional

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from genmol.data.base import MolDataset
from genmol.data.collate import pad_collate
from genmol.data.featurize import atom_types_to_onehot
from genmol.data.transforms import center_of_mass_zero, random_rotation_3d
from genmol.utils.chem import QM9_ATOMIC_NUMBERS
from genmol.utils.logging import get_logger

log = get_logger("genmol.data.qm9")

# Hoogeboom-EDM mirror, same URL PyG uses.
GDB9_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip"
GDB9_ALT_URL = "https://springernature.figshare.com/ndownloader/files/3195389"


def _download_gdb9(raw_dir: Path) -> Path:
    """If `gdb9.sdf` is not present, download it. Returns the SDF path."""
    sdf = raw_dir / "gdb9.sdf"
    if sdf.exists() and sdf.stat().st_size > 0:
        return sdf
    if sdf.exists():
        sdf.unlink()  # 0-byte leftover from a failed prior download
    raw_dir.mkdir(parents=True, exist_ok=True)
    log.info("Downloading QM9 gdb9.sdf to %s (~250 MB)", raw_dir)
    # Try the deepchem S3 zip first — figshare sits behind a WAF challenge that
    # urllib silently accepts as a 0-byte HTML response.
    zip_path = raw_dir / "qm9.zip"
    try:
        urllib.request.urlretrieve(GDB9_URL, str(zip_path))
        import zipfile

        with zipfile.ZipFile(zip_path) as z:
            z.extractall(raw_dir)
        zip_path.unlink()
    except Exception as e:
        log.warning("deepchem zip failed (%s); trying figshare mirror", e)
        if zip_path.exists():
            zip_path.unlink()
        urllib.request.urlretrieve(GDB9_ALT_URL, str(sdf))
    if not sdf.exists() or sdf.stat().st_size < 1_000_000:
        size = sdf.stat().st_size if sdf.exists() else 0
        raise RuntimeError(
            f"QM9 download produced an invalid SDF at {sdf} ({size} bytes). "
            "Both mirrors may be temporarily blocked — retry from a node with "
            "unrestricted egress."
        )
    return sdf


def _process_gdb9_to_tensors(sdf_path: Path, cache_path: Path) -> dict:
    """One-time SDF → (positions, atom_types) tensor cache."""
    from rdkit import Chem

    log.info("Processing %s — one-time SDF→tensor cache", sdf_path.name)
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=True)
    items = []
    skipped = 0
    for i, mol in enumerate(suppl):
        if mol is None:
            skipped += 1
            continue
        if mol.GetNumConformers() == 0:
            skipped += 1
            continue
        conf = mol.GetConformer()
        n = mol.GetNumAtoms()
        x = torch.tensor(
            [(conf.GetAtomPosition(j).x, conf.GetAtomPosition(j).y, conf.GetAtomPosition(j).z)
             for j in range(n)],
            dtype=torch.float32,
        )
        z = torch.tensor([mol.GetAtomWithIdx(j).GetAtomicNum() for j in range(n)], dtype=torch.long)
        items.append({"x": x, "z": z})
        if (i + 1) % 20_000 == 0:
            log.info("  %d / ~134k molecules processed", i + 1)

    log.info("Processed %d molecules (%d skipped)", len(items), skipped)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"items": items, "n_skipped": skipped}, cache_path)
    log.info("Cached → %s (%d items)", cache_path, len(items))
    return {"items": items}


class _QM9TorchDataset(MolDataset):
    """QM9 from a cached SDF processing pass. No PyG dependency."""

    def __init__(self, root: str, augment_rotation: bool = True):
        self.root = Path(root)
        self.augment = augment_rotation
        cache_path = self.root / "processed" / "qm9_genmol.pt"

        sdf = self.root / "raw" / "gdb9.sdf"
        if not sdf.exists():
            sdf = _download_gdb9(self.root / "raw")

        if not cache_path.exists():
            _process_gdb9_to_tensors(sdf, cache_path)

        blob = torch.load(cache_path, weights_only=False)
        self.items = blob["items"]
        self._atomic_numbers = QM9_ATOMIC_NUMBERS

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        x = item["x"].clone()
        h = atom_types_to_onehot(item["z"], self._atomic_numbers)
        mask = torch.ones(x.shape[0])
        if self.augment:
            x = random_rotation_3d(x)
        x = center_of_mass_zero(x, mask)
        return {"x": x, "h": h, "mask": mask}


class QM9DataModule(L.LightningDataModule):
    """Lightning glue around `_QM9TorchDataset` with subset/split support."""

    def __init__(
        self,
        root: str = "data/raw/qm9",
        batch_size: int = 64,
        num_workers: int = 4,
        subset_size: Optional[int] = None,
        val_fraction: float = 0.05,
        augment_rotation: bool = True,
        seed: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.subset_size = subset_size
        self.val_fraction = val_fraction
        self.augment_rotation = augment_rotation
        self.seed = seed

        self.train_ds: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None

    def prepare_data(self):
        # Trigger download + caching once per machine.
        _QM9TorchDataset(self.root, augment_rotation=False)

    def setup(self, stage: Optional[str] = None):
        full = _QM9TorchDataset(self.root, augment_rotation=self.augment_rotation)
        n = len(full)
        gen = torch.Generator().manual_seed(self.seed)
        if self.subset_size is not None and self.subset_size < n:
            n = self.subset_size
        indices = torch.randperm(len(full), generator=gen)[:n]
        n_val = max(1, int(n * self.val_fraction))
        val_idx = indices[:n_val].tolist()
        train_idx = indices[n_val:].tolist()
        self.train_ds = Subset(full, train_idx)
        self.val_ds = Subset(full, val_idx)

    def _loader(self, ds, shuffle: bool):
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=pad_collate,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self):
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_ds, shuffle=False)
