GOAL: Execute Stage 02 only: complexity measure calculation.

Read:
- `02_dnn/08_mnist/stages/02_complexity_measure/README.md`
- Stage 01 output `dataset_index.csv`

Input:
`02_dnn/08_mnist/runs/smoke/01_dataset_prepare/dataset_index.csv`

Compute graph total variation / NMSTV style complexity on standardized `X_train`.

Graph:
- multiscale kNN with k = [8,16,32]
- Euclidean distance
- symmetric graph by max weight
- weight: `exp(-dist^2/(2*sigma_k^2))`
- `sigma_k`: median nonzero kNN distance

Label TV:
\[
TV_k = \sum_{edges} w_{ij} 1[y_i\ne y_j] / \sum_{edges} w_{ij}.
\]

Balance baseline:
\[
baseline = 2p(1-p),\quad p=P(y=+1).
\]

\[
NMSTV_k = TV_k / max(baseline, 1e-12).
\]

Outputs:
`02_dnn/08_mnist/runs/smoke/02_complexity_measure/`
- `complexity_by_dataset.csv`
- `complexity_by_rule_summary.csv`
- `graph_stats_by_dataset_k.csv`
- `figures/fig01_nmstv_by_rule_boxplot.png`
- `figures/fig02_tv_by_k_rule.png`
- `figures/fig03_complexity_vs_label_balance.png`
- `REPORT.md`
- `run_config_resolved.json`
- `QC_STATUS.json`

Tests:
Create and run:
`02_dnn/08_mnist/tests/test_stage02_complexity_measure.py`

QC:
- all datasets have finite complexity;
- graph edge counts > 0;
- all three rules present;
- unexpected ordering is a warning, not a failure.

Stop after Stage 02.
