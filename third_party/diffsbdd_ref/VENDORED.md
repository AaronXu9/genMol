# Vendored: diffsbdd_ref

- **Upstream**: https://github.com/arneschneuing/DiffSBDD
- **License**: MIT (Schneuing et al. — verify on clone)
- **Status**: NOT YET FETCHED. Run `tools/crossdocked/fetch_diffsbdd_ref.sh` before M3.

## Plan

When fetched (M3), pin a specific commit, copy only the pocket-conditioned dynamics file (e.g. `dynamics.py`), and reimplement under `genmol/nn/pocket_egnn.py`. Do NOT copy their trainer or CLI.

Fill in this file after the cherry-pick with:
- Commit SHA pinned
- Files copied
- What we use / what we modified
