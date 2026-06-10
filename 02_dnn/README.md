# DNN Pipeline

This directory contains the synthetic-dataset and 3NN landscape pipeline. The
active public DNN results are the 18-beta runs:

- `18_beta_cell_30_dataset_30_reference`: production run with row-level shell
  raw payloads promoted under the public 18-beta path.
- `18_beta_cell_60_dataset_30_reference`: completed extension with compact
  shell/proxy summaries and figures retained.

The `18_beta_cell_30_dataset_30_reference` run is retained as the production
run with active dataset, reference, shell, proxy, and atlas paths. The
`18_beta_cell_60_dataset_30_reference` run is retained as a completed
summary-only extension.

## Stage Map

- `01_dataset_gen/`: 2D Ising synthetic datasets and active dataset indexes.
- `02_complexity_measure/`: NMSTV graph total-variation complexity summaries.
- `03_reference_search/`: 3NN training/reference search summaries.
- `04_sampling/`: PM-SAIS / adaptive CE-tempered SMC shell sampling.
- `05_proxy_local_entropy/`: compact `phi(d)` and phase-map tables/figures.
- `06_reference_atlas/`: reference-cloud geometry and straight-line barrier
  diagnostics for the 18-beta/30-dataset production run.
- `07_proxy_3D_landscape_for_visualize/`: optional browser visualization fitted
  to the stage-06 diagnostics.

## Active Path Pairs

Production run:

- dataset index:
  `01_dataset_gen/raw_outputs/18_beta_cell_30_dataset/dataset_index.csv`
- complexity:
  `02_complexity_measure/raw_outputs/18_beta_cell_30_dataset_nmstv/`
- reference summary:
  `03_reference_search/raw_outputs/18_beta_cell_30_dataset_30_reference/`
- reference-pool manifest:
  `04_sampling/raw_outputs/reference_pool/18_beta_cell_30_dataset_30_reference/`
- shell raw:
  `04_sampling/raw_outputs/shell_pool/18_beta_cell_30_dataset_30_reference/d_0.01_to_2.50_dense/`
- proxy:
  `05_proxy_local_entropy/raw_outputs/18_beta_cell_30_dataset_30_reference/d_0.01_to_2.50_dense/`
  and
  `05_proxy_local_entropy/figures/18_beta_cell_30_dataset_30_reference/d_0.01_to_2.50_dense/`

Completed extension:

- dataset index:
  `01_dataset_gen/raw_outputs/18_beta_cell_60_dataset/dataset_index.csv`
- reference-pool manifest:
  `04_sampling/raw_outputs/reference_pool/18_beta_cell_60_dataset_30_reference/`
- shell summary:
  `04_sampling/raw_outputs/shell_pool/18_beta_cell_60_dataset_30_reference/d_0.01_to_2.50_dense/`
- proxy:
  `05_proxy_local_entropy/raw_outputs/18_beta_cell_60_dataset_30_reference/d_0.01_to_2.50_dense/`
  and
  `05_proxy_local_entropy/figures/18_beta_cell_60_dataset_30_reference/d_0.01_to_2.50_dense/`
