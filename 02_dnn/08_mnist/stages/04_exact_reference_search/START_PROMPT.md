GOAL: Execute Stage 04 only: exact reference search.

Read:
- `02_dnn/08_mnist/runs/smoke/03_pool_design/POOL_CONTRACT.json`
- `02_dnn/08_mnist/stages/04_exact_reference_search/README.md`

Inputs:
- Stage 01 dataset index.
- Stage 03 model spec.

Smoke:
- selected_refs_per_dataset: 5
- max_attempts_per_dataset: 60
- optimizer: Adam warmup -> LBFGS polish
- full-batch preferred for reproducibility
- deterministic seeds

Reference acceptance:
\[
train\_error(	ilde	heta)=0.
\]

Store:
- theta vector
- CE_mean_train
- CE_sum_train
- CE_mean_test
- train/test error
- theta norm/norm_sq
- margin stats
- optimizer seed/attempt id
- selected policy

Outputs:
`02_dnn/08_mnist/runs/smoke/04_exact_reference_search/`
- `reference_index.csv`
- `selected_reference_pool/split_000/<rule>/ref_000/theta.npy`
- `selected_reference_pool/split_000/<rule>/ref_000/ref_summary.json`
- `attempt_logs/attempts.csv`
- `figures/fig01_reference_success_rate_by_rule.png`
- `figures/fig02_ref_ce_norm_scatter.png`
- `figures/fig03_margin_distribution_by_rule.png`
- `REPORT.md`
- `run_config_resolved.json`
- `QC_STATUS.json`

Tests:
Create and run:
`02_dnn/08_mnist/tests/test_stage04_reference_payload.py`

QC:
- 5 exact references per dataset if possible.
- all selected refs `train_error == 0`.
- theta length `P=3441`.
- no duplicate theta by L2 distance <= 1e-6.
- if insufficient refs, write blocked report or recommend max_attempt increase / backup architecture.

Stop after Stage 04.
