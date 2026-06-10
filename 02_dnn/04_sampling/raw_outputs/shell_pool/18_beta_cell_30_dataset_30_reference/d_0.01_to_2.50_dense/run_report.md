# 18-beta / 30-dataset dense shell sampling run

## Identity

- public run: `18_beta_cell_30_dataset_30_reference`
- selection rule: retained 18 beta cells with 30 datasets per cell
- selected beta cells: `0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.29, 0.31, 0.33, 0.35, 0.37, 0.39`
- selected cell count: `18`
- datasets per beta: `30`
- references per dataset: `30`
- shell range: `d_0.01_to_2.50_dense`
- attempted unit count: `4,050,000`

## Method

- backend: Python
- sampler: `exact_shell_l2_vmf_adaptive_ce_tempered_smc`
- sampler core: `02_dnn/04_sampling/src/pm_sais_core.py`
- model/loss code: `02_dnn/04_sampling/src/dnn_model.py`
- architecture: `2-48-48-1-tanh`, `P=2545`
- chunk size: `1024`
- shards: `32`
- device: `auto`

## Output Paths

- shell raw payloads:
  `02_dnn/04_sampling/raw_outputs/shell_pool/18_beta_cell_30_dataset_30_reference/d_0.01_to_2.50_dense/`
- proxy summaries:
  `02_dnn/05_proxy_local_entropy/raw_outputs/18_beta_cell_30_dataset_30_reference/d_0.01_to_2.50_dense/`
- proxy figures:
  `02_dnn/05_proxy_local_entropy/figures/18_beta_cell_30_dataset_30_reference/d_0.01_to_2.50_dense/`

## Smoke Estimate

- smoke report:
  `02_dnn/04_sampling/smoke_runs/36_30_30_dense_backend_bakeoff_20260524_193233/raw_outputs/python_8shards_chunk1024/logs/smoke_report.json`
- smoke completed: `16/16`, failed `0`
- measured rate: `1.344920` units/s
- projected subset elapsed: `836:28:51`

## Provenance Note

This directory is the active public shell-sampling path for the 18-beta /
30-dataset production run. Dataset and reference-pool config entries now point
to the materialized active 18-beta paths.
