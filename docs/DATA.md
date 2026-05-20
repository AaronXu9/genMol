# Datasets

## QM9 (Phase 1a)

- Auto-downloaded by PyG into `data/raw/qm9/` on first `QM9DataModule.prepare_data()`.
- ~134K molecules, ≤9 heavy atoms.
- Atom set: `{H, C, N, O, F}` (atomic numbers `[1, 6, 7, 8, 9]`).
- License: CC0 (research).

Trigger manually: `make prep-qm9`.

## CrossDocked2020 (Phase 1b)

- **NOT yet on disk.** Requires manual ~50 GB download from https://bits.csb.pitt.edu/files/crossdock2020/.
- Atom set: `{C, N, O, F, P, S, Cl, Br, I, H}` (matches DiffSBDD convention).
- We use the `pocket10` subset (10 Å pocket around each ligand).
- Split: DiffSBDD's `split_by_name.pt` for paper-comparable numbers.
- License: research only.

See `tools/crossdocked/download.sh` for instructions and `tools/crossdocked/extract_pockets.py` for the pocket extraction step (M3).

## PLINDER (Phase 1b alternative / extension)

- Located at `/mnt/katritch_lab2/aoxu/2024-06/v2/` (~121 GB on NFS).
- Real protein-ligand complexes from the PDB, curated by the PLINDER team.
- Symlink into the project tree via `make prep-plinder` (creates `data/raw/plinder` → mount).
- Subdirectories of note: `systems/`, `ligands/`, `splits/`, `index/`.
- License: per PLINDER paper / repository.

## Sample / generated molecule output

- Sampled molecules are written to `runs/<run_name>/samples/<step>.sdf`.
- Decoded via `genmol.sample.decode.decode_batch` — RDKit-based bond perception.
- For pocket-conditioned generation, the matched pocket PDB is symlinked alongside.

## Data directory layout (gitignored)

```
data/
├── raw/
│   ├── qm9/                  (auto-download from PyG)
│   ├── crossdocked/          (manual ~50 GB)
│   └── plinder/  -> /mnt/katritch_lab2/aoxu/2024-06/v2/
├── processed/
│   ├── qm9/
│   ├── crossdocked/
│   └── plinder/
└── splits/                   (mirrored from genmol/data/splits/ at runtime)
```

The only data committed to git is `genmol/data/splits/*.json` (small frozen index files).
