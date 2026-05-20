"""Build a pickle of canonical train SMILES for novelty computation.

Reads gdb9.sdf, removes Hs, canonicalizes, dedupes, writes a pickle to
`data/processed/qm9/train_smiles.pkl`. Run once per machine.
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def main(sdf_path: str = "data/raw/qm9/raw/gdb9.sdf",
         out_path: str = "data/processed/qm9/train_smiles.pkl") -> None:
    sdf = Path(sdf_path)
    out = Path(out_path)
    if not sdf.exists():
        print(f"ERROR: {sdf} not found. Run prepare_qm9 first.", file=sys.stderr)
        sys.exit(1)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading {sdf} (~225 MB, ~134K molecules)")
    t0 = time.time()
    suppl = Chem.SDMolSupplier(str(sdf), removeHs=False, sanitize=True)
    smiles: set[str] = set()
    skipped = 0
    for i, mol in enumerate(suppl):
        if mol is None:
            skipped += 1
            continue
        try:
            # Decoder produces mols with explicit H. We canonicalize by
            # removing H to get the same string regardless of H representation.
            m = Chem.RemoveHs(mol)
            smiles.add(Chem.MolToSmiles(m))
        except Exception:
            skipped += 1
        if (i + 1) % 30_000 == 0:
            print(f"  {i+1} read, {len(smiles)} unique so far")

    with open(out, "wb") as f:
        pickle.dump(smiles, f)
    dt = time.time() - t0
    print(f"DONE in {dt:.1f}s: {len(smiles)} unique canonical SMILES ({skipped} skipped) → {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
