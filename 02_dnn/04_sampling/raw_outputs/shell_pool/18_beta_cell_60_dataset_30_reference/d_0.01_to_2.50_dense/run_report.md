# 18/60/30 dense sampling run report

## Backend

- selected backend: `python`
- Python runner: sharded wrapper around active `02_dnn/04_sampling/src/pm_sais_core.py::sample_unit`
- allowed physical GPUs: `2,3`

## Full Run

- target: `18_beta_cell_60_dataset_30_reference/d_0.01_to_2.50_dense`
- unit count: `8,100,000`
- shards: `32`
- chunk size: `1024`
- device: `auto`
- projected baseline from completed 18/30/30 dense: `129:46:25`, before live ETA correction
- raw storage estimate: `~900 GB+`

## Output Paths

- raw: `02_dnn/04_sampling/raw_outputs/shell_pool/18_beta_cell_60_dataset_30_reference/d_0.01_to_2.50_dense`
- QC: `02_dnn/04_sampling/QC/shell_pool/18_beta_cell_60_dataset_30_reference/d_0.01_to_2.50_dense`
- sampling figures: `02_dnn/04_sampling/figures/18_beta_cell_60_dataset_30_reference/d_0.01_to_2.50_dense`
- proxy figures: `02_dnn/05_proxy_local_entropy/figures/18_beta_cell_60_dataset_30_reference/d_0.01_to_2.50_dense`

## Progress

- aggregate status: `02_dnn/04_sampling/raw_outputs/shell_pool/18_beta_cell_60_dataset_30_reference/d_0.01_to_2.50_dense/logs/aggregate_status.json`
- aggregate progress log: `02_dnn/04_sampling/raw_outputs/shell_pool/18_beta_cell_60_dataset_30_reference/d_0.01_to_2.50_dense/logs/aggregate_progress.jsonl`
- stage progress latest JSON: `99_codex_bin/parallel_sampling_18_100_30/progress/latest.json`
- stage progress latest MD: `99_codex_bin/parallel_sampling_18_100_30/progress/latest.md`
- shard logs: `02_dnn/04_sampling/raw_outputs/shell_pool/18_beta_cell_60_dataset_30_reference/d_0.01_to_2.50_dense/shards`
