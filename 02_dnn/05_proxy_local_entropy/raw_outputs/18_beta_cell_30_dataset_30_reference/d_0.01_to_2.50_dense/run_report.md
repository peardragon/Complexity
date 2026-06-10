# Proxy local entropy report: 18_beta_cell_30_dataset_30_reference / d_0.01_to_2.50_dense

This directory stores the compact proxy tables for the active 18-beta /
30-dataset production run.

## Method

- input shell run:
  `02_dnn/04_sampling/raw_outputs/shell_pool/18_beta_cell_30_dataset_30_reference/d_0.01_to_2.50_dense/`
- proxy method: full regularized local entropy view using full, stripped, and
  reference-prior-corrected shell logZ fields when available
- area term: L2 shell volume contribution in `P=2545`
- derivative view: radial derivatives summarized by beta and radius
- H-threshold view: phase-map table from retained `R_H` summaries

## Retained Tables

- `summary_tables/absolute_phi_by_beta_radius.csv`
- `summary_tables/delta_phi_by_beta_radius.csv`
- `summary_tables/dphi_dr_by_beta_radius.csv`
- `summary_tables/hq_by_beta_radius.csv`

## Figure Root

`02_dnn/05_proxy_local_entropy/figures/18_beta_cell_30_dataset_30_reference/d_0.01_to_2.50_dense/`

## Provenance Note

This public path is the active proxy-local-entropy summary for the 18-beta /
30-dataset production run. Its upstream shell, dataset, and reference-pool
inputs are materialized under active 18-beta paths.
