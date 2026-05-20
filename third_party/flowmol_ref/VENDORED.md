# Vendored: flowmol_ref

- **Upstream**: https://github.com/Dunni3/FlowMol
- **License**: MIT (verify on clone)
- **Status**: NOT YET FETCHED. Run `tools/vendor_flowmol.sh` before M2.

## Plan

When fetched (M2), pin a specific commit and use as a reference for the flow-matching loss formulation and Euler integrator. We reimplement in `genmol/process/flow_matching.py`. We do not import their model — our backbone is EGNN.

Fill in this file after retrieval with:
- Commit SHA pinned
- Files referenced (no verbatim copy, just notes)
