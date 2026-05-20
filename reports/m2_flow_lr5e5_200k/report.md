# Benchmark report: FLOW
- **Checkpoint**: `logs/qm9_flow_lr5e5-20260520-135304/checkpoints/epoch=102-step=200000.ckpt`
- **N samples**: 512
- **Sampling steps**: 100
- **N atoms**: random uniform [11, 22]
- **Sampling time**: 145.0s (3.53 mol/s)

## RDKit 2D metrics
- **total**: 512.0000
- **valid**: 207.0000
- **validity**: 0.4043
- **uniqueness**: 1.0000
- **novelty**: 1.0000
- **mean_qed**: 0.3819

## 3D geometry
- **mean_bond_length**: 1.2674
- **std_bond_length**: 0.2430
- **n_bonds**: 3227

## Example valid SMILES
- `[H][C+]1C([H])([H])[C-]([N-][C-]2C([H])([C-]([H])[O-])[N+]2([H])[H])C1([H])[H]`
- `[H][C-][C]([H])[N-]C1([H])[C]2[C]([H])[C+]1[C+]2[C-][H]`
- `[H+].[H][C-]1[C@]([H])(C([H])([H])[H])[C-]([H])[C@@]1([H])C([H])([H])[O+]=O`
- `[H][N-]C(=O)[C+]1C([H])([H])[C@]1([H])[N+]#[C-]`
- `[H][N][C][N][C]1[C]([H])C12[N]C2([H])O[H]`
- `N#C[O-].[H][N+]#C/C([H])=C(/[H])F`
- `[H][C]F.[H][N]C([H])([H])[C]([H])[N]`
- `[H][C]OC1([O+][O+][H])OC1([H])[C-]([H])[H]`
- `[H]OC([H])([C][C+]1ON1[H])C([H])([H])[C+]([H])[C-]([H])C([H])([H])[H]`
- `[H]/[O+]=[N+]1/[N+](=N\[C@@]2([O-])N=C2[O-])[C@@]1([H])[O-]`
- `[H][C]([O])C([H])([H])C1(C([H])([H])[N]C([H])([H])[H])[C]([H])C1([H])[C]([H])[H]`
- `[H]OC1([C-]([H])[O-])C([H])([H])C12C([H])([H])[N+]2([H])[H]`
- `[H+].[H][C-]([H])[N@+]12C(=O)[N-][N@@+]1([H])[C-]2[H]`
- `[H][C][N-][C+]([H])[C-]([H])[O-].[H][N-][C-]([H])[C][C+]([H])[H]`
- `[H+].[H][C-][C]([H])[C+]1[C]C([H])(O[H])[C-]1`
- `[H].[H]OC1([H])[N]N1[C]([N])[C][O]`
- `[H][C][C]([H])[H].[H][N][C]([O])O[H].[N][C]([O])[O]`
- `[H+].[H][C+]1N2C([H])([H])C12[C][C-]([H])C([H])([H])[H]`
- `[H][C+]([O-])[N+]([H])([H])C1([H])[C+]([c-]2[n-]o2)[C-]1N([H])[H]`
- `[H][C-]1C([H])([O-])OC23N([H])C2([H])C13[H]`