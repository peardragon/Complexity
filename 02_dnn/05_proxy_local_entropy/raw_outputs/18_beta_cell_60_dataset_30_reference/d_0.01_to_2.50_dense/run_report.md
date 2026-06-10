# Proxy local entropy report: 18_beta_cell_60_dataset_30_reference / d_0.01_to_2.50_dense

This directory stores compact proxy tables for the completed 18-beta /
60-dataset extension.

## Method

- input shell summary:
  `02_dnn/04_sampling/raw_outputs/shell_pool/18_beta_cell_60_dataset_30_reference/d_0.01_to_2.50_dense/summary_tables/`
- proxy method: full regularized local entropy view using full, stripped, and
  reference-prior-corrected shell logZ fields
- area term: L2 shell volume contribution in `P=2545`
- q values for H-threshold maps: `0.5`, `0.9`, `0.99`
- accuracy quantile table: not regenerated in this upload pass because it
  requires rereading every retained sample payload NPZ

## Retained Tables

- `summary_tables/absolute_phi_by_beta_radius.csv`
- `summary_tables/delta_phi_by_beta_radius.csv`
- `summary_tables/dphi_dr_by_beta_radius.csv`
- `summary_tables/high_beta_curve_comparison.csv`
- `summary_tables/high_beta_trend_by_radius.csv`
- `summary_tables/hq_by_beta_radius.csv`

## Acceptance Snapshot

- sampling rows: `8,100,000`
- failed units: `0`
- proxy rows: `4,500` each for absolute, delta, and derivative tables
- high-beta threshold: `beta >= 0.29`

The corresponding acceptance report is
`02_dnn/04_sampling/raw_outputs/shell_pool/18_beta_cell_60_dataset_30_reference/d_0.01_to_2.50_dense/final_acceptance_and_high_beta_report.md`.
