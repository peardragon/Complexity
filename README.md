# Complexity

This repository contains the retained theory and 3NN pipeline for studying how
dataset complexity changes the local geometry of neural-network solution
landscapes.

## Research Question

The working hypothesis is that the geometry of a trained model's loss landscape
is affected not only by overparameterization, but also by the complexity of the
dataset itself. Dataset complexity is measured from the ground-truth labels on a
graph; landscape geometry is observed through local shell entropy and related
Franz-Parisi style proxy quantities around trained solutions.

## Pipeline

1. Define a dataset-complexity observable: normalized multiscale total
   variation (NMSTV), a graph total-variation/stiffness measure of the label
   decision boundary.
2. Validate the shell-entropy sampler on the perceptron with L2 regularization,
   where analytic theory is available.
3. Generate 2D synthetic datasets from Ising spin configurations over beta
   cells. Higher beta-cell structure changes the geometry of same-label
   regions.
4. Train a fixed 3NN architecture (`2-48-48-1-tanh`, `P=2545`) and select
   exact or sampling-eligible reference solutions.
5. Estimate local shell statistics using the DNN PM-SAIS / adaptive
   CE-tempered SMC sampler on L2 shells around each reference.
6. Convert shell summaries into compact proxy-local-entropy tables and figures:
   `phi(d)`, energetic/entropic terms, radial derivatives, and H-threshold
   phase maps.
7. Compare how these curves change with dataset complexity. The retained claim
   is continuous deformation of the local-entropy profile over the active beta
   sweep, not a critical transition claim.

## Active Layout

```text
D:\Complexity
|-- 01_theory
|   |-- 01_theory_analytic
|   |-- 02_theory_sampling
|   `-- 03_theory_comparison
|-- 02_dnn
|   |-- 01_dataset_gen
|   |-- 02_complexity_measure
|   |-- 03_reference_search
|   |-- 04_sampling
|   |-- 05_proxy_local_entropy
|   |-- 06_reference_atlas
|   `-- 07_proxy_3D_landscape_for_visualize
|-- 99_backup
`-- 99_codex_bin
```

Each retained stage uses `config/`, `src/`, `raw_outputs/`, `figures/`,
optional `QC/`, and `smoke_runs/` where applicable.

## Active Finals

Theory validation final:

- analytic output:
  `01_theory/01_theory_analytic/raw_outputs/theory_full_rs_alpha0p1.csv`
- sampling pools:
  `01_theory/02_theory_sampling/raw_outputs/reference_pool/` and
  `01_theory/02_theory_sampling/raw_outputs/shell_pool/`
- comparison figure:
  `01_theory/03_theory_comparison/figures/fig00_dense_qc_N_convergence_alpha0p1.png`

DNN production final:

- public run identity: `18_beta_cell_30_dataset_30_reference`
- raw shell payloads:
  `02_dnn/04_sampling/raw_outputs/shell_pool/18_beta_cell_30_dataset_30_reference/d_0.01_to_2.50_dense/`
- proxy summary and figures:
  `02_dnn/05_proxy_local_entropy/raw_outputs/18_beta_cell_30_dataset_30_reference/d_0.01_to_2.50_dense/`
  and
  `02_dnn/05_proxy_local_entropy/figures/18_beta_cell_30_dataset_30_reference/d_0.01_to_2.50_dense/`
- reference-cloud diagnostics:
  `02_dnn/06_reference_atlas/raw_outputs/18_beta_cell_30_dataset_30_reference/d_0.01_to_2.50_dense/`

DNN completed extension:

- run identity: `18_beta_cell_60_dataset_30_reference`
- selected beta cells: the same 18 beta cells as the production final
- datasets per beta: 60
- reference/shell/proxy status: compact summary tables and figures are
  retained; theta and row-level shell sample payloads are not retained in this
  path
- acceptance: 8,100,000 sampling rows, 0 failed units, 4,500 proxy rows per
  proxy table family

## Legacy Boundary

Superseded 9-beta/10-dataset outputs and non-active 36-beta dense outputs were
moved to `99_backup/cleanup_20260610_141822/`. The cleanup manifest records the
rollback map. A large temporary transfer archive remains at
`02_dnn/transfer_archives/` because Windows reported the `.tmp` file as held by
another process during cleanup; this is recorded as a blocked move in the same
manifest and is not an active research output.

Large binary payloads remain intentionally untracked by Git. See
`RAW_PAYLOADS.md` for the payload policy. See `FIGURE_ANALYSIS_INDEX.md` for
the retained analysis/QC/review figure map.
