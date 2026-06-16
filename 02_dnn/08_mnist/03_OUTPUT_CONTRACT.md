# 03_OUTPUT_CONTRACT

## Global naming

Experiment ID:

```text
mnist14_3rule_1024
```

Rules:

```text
real_even_odd
teacher_nn
random_label
```

Main architecture:

```text
196-16-16-1-tanh
P=3441
```

## Smoke paths

```text
02_dnn/08_mnist/runs/smoke/00_repo_audit/
02_dnn/08_mnist/runs/smoke/01_dataset_prepare/
02_dnn/08_mnist/runs/smoke/02_complexity_measure/
02_dnn/08_mnist/runs/smoke/03_pool_design/
02_dnn/08_mnist/runs/smoke/04_exact_reference_search/
02_dnn/08_mnist/runs/smoke/05_pool2_pm_sais_sampling/
02_dnn/08_mnist/runs/smoke/06_results_figures/
02_dnn/08_mnist/runs/smoke/final_report/
```

## Required stage outputs

### Stage 01 dataset

```text
dataset_index.csv
raw_datasets/split_000/<rule>/dataset.npz
metadata/split_summary.csv
metadata/label_balance_summary.csv
figures/fig01_mnist_28_vs_14_montage.png
```

### Stage 02 complexity

```text
complexity_by_dataset.csv
complexity_by_rule_summary.csv
graph_stats_by_dataset_k.csv
figures/fig01_nmstv_by_rule_boxplot.png
```

### Stage 04 references

```text
reference_index.csv
selected_reference_pool/split_000/<rule>/ref_000/theta.npy
selected_reference_pool/split_000/<rule>/ref_000/ref_summary.json
attempt_logs/attempts.csv
```

### Stage 05 PM-SAIS

```text
selected_lambda.json
lambda_selection_report.csv
shell_summary_by_unit.csv
shell_summary_by_rule_radius.csv
qc_by_rule_radius.csv
```

### Stage 06 final figures

```text
phi_by_rule_radius.csv
phi_bootstrap_by_rule_radius.csv
qc_pass_by_rule_radius.csv
complexity_reference_sampling_joined.csv
final_claim_table.csv
figures/fig04_phi_energy_three_rules_main.png
figures/fig05_phi_full_three_rules.png
figures/fig07_sampling_qc_pass_heatmap.png
REPORT.md
```
