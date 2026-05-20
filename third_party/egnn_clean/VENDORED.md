# Vendored: egnn_clean

- **Upstream**: https://github.com/vgsatorras/egnn
- **Local source**: `/home/aoxu/projects/egnn/models/egnn_clean/`
- **License**: MIT (Victor Garcia Satorras, 2021) — see `LICENSE`
- **Retrieval date**: 2026-05-16
- **Method**: file copy from local clone (no upstream commit pinned because copied from on-disk working copy)

## Files copied

- `egnn_clean.py` — `E_GCL`, `EGNN`, `unsorted_segment_sum`, `unsorted_segment_mean`, `get_edges`, `get_edges_batch`.
- `__init__.py`

## What we use

- `E_GCL` (single equivariant graph-conv layer) and `EGNN` (stack of E_GCL).
- Wrapped (NOT subclassed) by `genmol/nn/egnn.py:EGNNBackbone`.

## Modifications

**None.** Files are verbatim. Any future modification MUST be recorded here with a commit SHA delta.

## Why vendored rather than pip-installed

Upstream is not on PyPI. The repo provides reference code, not a package. Vendoring lets us pin the implementation we trained against.
