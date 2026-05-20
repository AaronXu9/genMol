"""Run the full eval suite on samples produced from a checkpoint."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from genmol.eval.benchmarks import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=False, help="Lightning ckpt path")
    p.add_argument("--samples", type=str, required=False, help="SDF of generated mols")
    p.add_argument("--output-dir", type=str, default="reports/eval")
    args = p.parse_args()

    if args.samples is None and args.ckpt is None:
        raise SystemExit("Provide either --samples (SDF) or --ckpt (Lightning checkpoint)")

    if args.samples is not None:
        from rdkit import Chem
        suppl = Chem.SDMolSupplier(args.samples, sanitize=True)
        mols = [m for m in suppl]
    else:
        raise NotImplementedError(
            "M2+: load ckpt → sample → evaluate. For now, pass --samples directly."
        )

    metrics = evaluate(mols, output_dir=args.output_dir)
    print(metrics)


if __name__ == "__main__":
    main()
