# Benchmark report: EDM
- **Checkpoint**: `logs/qm9_edm-20260519-200717/checkpoints/epoch=102-step=200000.ckpt`
- **N samples**: 512
- **Sampling steps**: 250
- **N atoms**: random uniform [11, 22]
- **Sampling time**: 849.5s (0.60 mol/s)

## RDKit 2D metrics
- **total**: 512.0000
- **valid**: 461.0000
- **validity**: 0.9004
- **uniqueness**: 1.0000
- **novelty**: 1.0000
- **mean_qed**: 0.4162

## 3D geometry
- **mean_bond_length**: 1.2582
- **std_bond_length**: 0.1941
- **n_bonds**: 7777

## Example valid SMILES
- `[H][C+]1[C-]([H])[C+]([H])C([H])([H])N1C([H])([H])C([H])([H])[C-]([H])[O-]`
- `[H]C#C[C@@]12C([H])=[N+]=C([H])[C@]1([H])[C@]2([H])[O-]`
- `[H][C+]1OC([H])([H])[C-](C([H])([H])[H])C([H])([H])[C-]1[H]`
- `[H]C([H])([H])[N+]1=C(C#N)O[N-]O1`
- `[H]N1[C@]2([H])C(=O)[C@@]([H])(C2([H])[H])[C@@]2([H])C([H])([H])[C@@]12[H]`
- `[H][C][C]C1([H])C([H])([H])C1([H])[N-][C+]([H])[O-]`
- `[H]C([H])([H])C([H])([H])[N]C([H])([H])C1([H])[C]([O])C1([H])[H]`
- `[H]C([H])([H])C12OC1(C([H])([H])[H])[N+]21C([H])([H])C1([H])[H]`
- `[H][N-][C+]([O-])[C+]1C2([C][N-2])N([H])[N+]12[H]`
- `[H][N-][C-]1O[C+](C([H])([H])[H])[C+]([N-][O-])[C+]1[H]`
- `[H][C+]([H])[C@]1([H])C([H])([H])C([O-])=C(C#N)C1([H])[H]`
- `[H][C]([N])[N]N1[C]([N][C]([O])[O])C1([H])[H]`
- `[H][C-]([O-])C1([H])[C+]([C][N-2])C1([H])[H]`
- `[H]O[C@@]([H])(/C(=[O+]\C([H])([H])C([H])([H])[H])C([H])([H])[H])C([H])([H])[O-]`
- `[H][C+]([O-])C1([H])C([H])([H])[C-]([H])C([H])([H])C2([H])N([H])C12[H]`
- `[H][C]([C][N])[C]([N])N([H])[C]([H])F`
- `[H]c1nnc([H])n1C([H])([H])[H]`
- `[H][C+]1[C-]([C+]([O-])N([H])[H])O[C-]([H])N1[H]`
- `[H]O[C@@]([H])(C(C(=O)C([H])([H])C([H])([H])[H])=[N+]([H])[H])[C-]([H])[H]`
- `[H][c+]1[n-]n([H])[n-][c+]1C([H])([O-])[C-]([H])[O-]`