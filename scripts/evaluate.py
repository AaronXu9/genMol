"""Run the full eval suite. Thin wrapper around scripts/run_benchmark.py.

For maximum control use run_benchmark.py directly. This script is the
default `make eval CKPT=...` target.

Usage:
    python scripts/evaluate.py --ckpt PATH --model {edm,flow} [other args]

If --model is omitted, infers from the checkpoint's stored hyperparameters
when possible.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model", choices=["edm", "flow"], default=None,
                   help="if omitted, infer from ckpt path")
    p.add_argument("--n-samples", type=int, default=512)
    p.add_argument("--num-steps", type=int, default=None,
                   help="defaults: edm=250, flow=100")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--output-dir", default=None,
                   help="defaults to reports/<ckpt-stem>")
    args = p.parse_args()

    model = args.model
    if model is None:
        if "edm" in args.ckpt.lower() or "diffusion" in args.ckpt.lower():
            model = "edm"
        elif "flow" in args.ckpt.lower() or "fm" in args.ckpt.lower():
            model = "flow"
        else:
            raise SystemExit("Cannot infer --model from ckpt path; pass --model explicitly")

    num_steps = args.num_steps if args.num_steps is not None else (250 if model == "edm" else 100)
    output_dir = args.output_dir or f"reports/{Path(args.ckpt).parent.parent.name}"

    cmd = [
        sys.executable, str(_REPO / "scripts" / "run_benchmark.py"),
        "--ckpt", args.ckpt,
        "--model", model,
        "--n-samples", str(args.n_samples),
        "--num-steps", str(num_steps),
        "--batch", str(args.batch),
        "--output-dir", output_dir,
    ]
    print(f"$ {' '.join(cmd)}")
    raise SystemExit(subprocess.call(cmd, cwd=str(_REPO)))


if __name__ == "__main__":
    main()
