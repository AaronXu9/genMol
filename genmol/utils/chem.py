"""Chemistry utilities: atom-type maps, valence sanitizers, SMILES helpers."""
from __future__ import annotations

# QM9 atom types in canonical order. EDM and FlowMol both use this set.
QM9_ATOM_TYPES = ["H", "C", "N", "O", "F"]
QM9_ATOMIC_NUMBERS = [1, 6, 7, 8, 9]
QM9_ATOM_TYPE_TO_IDX = {t: i for i, t in enumerate(QM9_ATOM_TYPES)}

# CrossDocked ligand atom types (10 elements, matches DiffSBDD).
CROSSDOCKED_ATOM_TYPES = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "H"]
CROSSDOCKED_ATOMIC_NUMBERS = [6, 7, 8, 9, 15, 16, 17, 35, 53, 1]

# CrossDocked pocket residue atom types (typically all heavy elements + H).
POCKET_ATOM_TYPES = ["C", "N", "O", "S", "H"]


def atom_types_for_dataset(name: str) -> list[str]:
    if name == "qm9":
        return QM9_ATOM_TYPES
    if name == "crossdocked":
        return CROSSDOCKED_ATOM_TYPES
    raise ValueError(f"Unknown dataset: {name}")
