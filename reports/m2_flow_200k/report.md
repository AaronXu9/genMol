# Benchmark report: FLOW
- **Checkpoint**: `logs/qm9_flow-20260520-054542/checkpoints/epoch=102-step=200000.ckpt`
- **N samples**: 512
- **Sampling steps**: 100
- **N atoms**: random uniform [11, 22]
- **Sampling time**: 289.7s (1.77 mol/s)

## RDKit 2D metrics
- **total**: 512.0000
- **valid**: 314.0000
- **validity**: 0.6133
- **uniqueness**: 1.0000
- **novelty**: 1.0000
- **mean_qed**: 0.3727

## 3D geometry
- **mean_bond_length**: 1.2472
- **std_bond_length**: 0.2193
- **n_bonds**: 5081

## Example valid SMILES
- `[H][C+]1[C-]([H])[C+]([H])C([H])([H])N1C([H])([H])C([H])([H])[C-]([H])[O-]`
- `[H]C([H])([H])O[C+]([O-])N1C([H])([H])C1([H])[H].[H][C-]([H])[H]`
- `[H][N-][C+]([O-])[C+]1C([H])([C][N-2])[N+]1([H])[H]`
- `[H][C+]1[C-](OC2([H])C([H])([H])C23[N-]O3)N1[H]`
- `[H]C1=C([O-])[N+]1=C=C=[N-].[H][C+]1C([H])([H])C1([H])[H]`
- `[H][C][C]1[C]([N][C]([H])[C][O])N1[C]([H])[N]`
- `[H][C-][C+]([C]([H])[C]([H])[H])C([H])([H])[C][C][N]`
- `[H][C+]([H])[C+](C([H])([H])[H])C1([H])N([H])C1([H])[C-]([H])[C-]([H])[O-]`
- `[H]/N=C(/[O-])[C-]([H])F.[H]C1=[N+]=[O+]1`
- `[H]C([H])([H])[N]C1([H])[N]N1[C]F`
- `[H]C([H])=[O+]C(=O)/C([O-])=C(/[H])N([H])[H]`
- `[H]O[C]([C]([H])N([H])[H])C([H])([H])[H].[H][N]C([H])([H])[C]([H])[O]`
- `[H][C]1C2([O])N(N3C4([H])[N]C134)C2([H])[O]`
- `[H]O[C]1[C]([C]([H])[H])N(C([H])([H])[H])[C]([N])N1[H]`
- `N#C[O-].[H]C([H])=[N+]=C([H])N([H])[H]`
- `[H][C+]([H])[C][C]N([H])[H].[H][C-]([O-])[C]C([H])([H])[H]`
- `[H][C-]1C([H])([H])[C@@]2([H])[C@@]13O[N@+]23[H]`
- `[H]/[O+]=C1/C([H])([H])C12C#C2.[H]C#[C-]`
- `[H][O+]1C2([H])C([H])([H])C12[c-]1[n-]o1`
- `[H][C+](OC([H])([H])[H])C([H])([H])[C-]1C([H])([H])C1([H])C([H])([H])[C-]([H])[H]`