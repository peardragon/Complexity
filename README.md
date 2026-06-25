# Complexity

This repository keeps only the promoted theory and 3NN artifacts needed to
inspect and reproduce the current final figures.

## Active Layout

```text
D:\Complexity
├─ 01_theory
│  ├─ 01_theory_analytic
│  ├─ 02_theory_sampling
│  └─ 03_theory_comparison
├─ 02_dnn
│  ├─ 01_dataset_gen
│  ├─ 02_complexity_measure
│  ├─ 03_reference_search
│  ├─ 04_sampling
│  └─ 05_proxy_local_entropy
├─ 99_backup
└─ 99_codex_bin
```

Each retained stage uses:

- `config/` for active default and smoke config templates.
- `src/` for code retained for the active stage.
- `raw_outputs/` for promoted raw data when the stage produces data.
- `figures/` for promoted figures.
- `QC/` for promoted sampling QC outputs when applicable.
- `smoke_runs/` for smoke checks, short experiments, and temporary test beds.

Legacy `summary/`, `results_summary/`, `results_raw/`, and DNN `d2/`
directories are archived in `99_backup/cleanup_<timestamp>/old_workspace/`.
Do not create new active final outputs using that legacy contract.

## Promoted Theory Final

The theory-side final is based on
`two_pool_perceptron_alpha0p1_dense_qc_full_split_pass_merged`.

- Analytic data:
  `01_theory/01_theory_analytic/raw_outputs/theory_full_rs_alpha0p1.csv`
- Sampling reference pool:
  `01_theory/02_theory_sampling/raw_outputs/reference_pool/`
- Sampling shell pool:
  `01_theory/02_theory_sampling/raw_outputs/shell_pool/`
- Combined final figure:
  `01_theory/03_theory_comparison/figures/fig00_dense_qc_N_convergence_alpha0p1.png`

The promoted sampling provenance is 16,800 final sample units from the base
`10x10_N40_80_160_320_p2048_r0p05`, far split `N40/N80/N160/N320_p32768`,
and 15 high-particle replacement units.

## Promoted 3NN Final

The 3NN production final is:

```text
18_beta_cell_90_dataset_30_reference/d_0.01_to_2.50_dense
```

The active user-facing figure root is:

```text
02_dnn/05_proxy_local_entropy/figures/18_beta_cell_90_dataset_30_reference/d_0.01_to_2.50_dense/
```

The corresponding retained production roots are:

- `02_dnn/01_dataset_gen/raw_outputs/18_beta_cell_90_dataset/`
- `02_dnn/02_complexity_measure/raw_outputs/18_beta_cell_90_dataset_30_reference/`
- `02_dnn/03_reference_search/raw_outputs/18_beta_cell_90_dataset_30_reference/`
- `02_dnn/04_sampling/raw_outputs/reference_pool/18_beta_cell_90_dataset_30_reference/`
- `02_dnn/04_sampling/raw_outputs/shell_pool/18_beta_cell_90_dataset_30_reference/d_0.01_to_2.50_dense/`
- `02_dnn/05_proxy_local_entropy/raw_outputs/18_beta_cell_90_dataset_30_reference/d_0.01_to_2.50_dense/`

## Smoke Runs

Every stage has a `smoke_runs/` directory. Use it for smoke checks, short
exploratory runs, and temporary test beds. A smoke run repeats the active stage
shape under a user-friendly name, for example:

```text
smoke_runs/smoke_default_20260523/config
smoke_runs/smoke_default_20260523/src
smoke_runs/smoke_default_20260523/raw_outputs
smoke_runs/smoke_default_20260523/figures
```

Smoke outputs are not final until explicitly promoted into the stage's active
`raw_outputs/`, `figures/`, or `QC/` path with provenance.

## Rollback

Cleanup runs write a manifest to:

```text
99_backup/cleanup_<timestamp>/cleanup_manifest.json
```

Non-retained active paths are moved, not permanently deleted. Rollback is a
path-map restore from that manifest.
