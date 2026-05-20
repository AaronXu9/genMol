# Vendored third-party components

Each subdirectory contains code copied from external repositories, kept under their original license. We vendor rather than pip-install for: (1) frozen reproducibility, (2) ability to cherry-pick model code without inheriting an entire training framework, (3) avoiding dependency conflicts with our `genmol` env.

| Component | Source | License | Purpose |
|---|---|---|---|
| `egnn_clean/` | https://github.com/vgsatorras/egnn (Victor Garcia Satorras, 2021) | MIT | E(n)-equivariant network primitives (`E_GCL`, `EGNN`) — used as the backbone of every model |
| `diffsbdd_ref/` | https://github.com/arneschneuing/DiffSBDD (Schneuing et al.) | MIT (verify on clone) | **Reference only.** Pocket-conditioned EGNN dynamics formulation. We reimplement under `genmol/nn/pocket_egnn.py` |
| `flowmol_ref/` | https://github.com/Dunni3/FlowMol | MIT (verify on clone) | **Reference only.** Flow-matching loss formulation + Euler integrator for molecular generation |

Each subdir includes:
- `LICENSE` — copied from upstream, unmodified.
- `VENDORED.md` — upstream URL, commit SHA, retrieval date, files copied, what we use, what we modified.

When vendoring new code: always preserve `LICENSE`, fill out `VENDORED.md` truthfully, and update this index.
