# Reference Cloud Proxy Metrics

This stage computes reference-cloud diagnostics for the active 18-beta /
30-dataset production run.

## Method

- input reference pool:
  `04_sampling/raw_outputs/reference_pool/18_beta_cell_30_dataset_30_reference/`
- dataset source:
  `01_dataset_gen/raw_outputs/18_beta_cell_30_dataset/`
- metrics:
  - `S_ref`: RMS pairwise normalized reference spread
  - `Q_ref`: mean pairwise cosine alignment
  - `H_err`, `H_CE`: mean-center defect / hollow diagnostics
  - `B_lin`: straight-line interpolation barrier proxy
- code: `src/compute_reference_cloud_metrics.py`

## Active Outputs

- `raw_outputs/18_beta_cell_30_dataset_30_reference/d_0.01_to_2.50_dense/`
- `figures/18_beta_cell_30_dataset_30_reference/d_0.01_to_2.50_dense/`
- `QC/18_beta_cell_30_dataset_30_reference/d_0.01_to_2.50_dense/`

## Claim Boundary

`B_lin` is a straight-line CE/error barrier diagnostic only. It is not a proof
of nonlinear solution connectivity, and this stage does not compute graph
connectivity metrics such as `G_tau`.
