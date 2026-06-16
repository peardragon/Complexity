GOAL: Execute Stage 05 only: Pool2 PM-SAIS sampling.

Read:
- `02_dnn/08_mnist/01_FORMULAS_PM_SAIS.md`
- `02_dnn/08_mnist/runs/smoke/03_pool_design/POOL_CONTRACT.json`
- `02_dnn/08_mnist/runs/smoke/04_exact_reference_search/reference_index.csv`
- this stage README

Main estimator:
PM-SAIS \(H=\infty\).

Optional diagnostics:
PM-RLI \(H\in\{8,4,2\}\), but main claim remains \(H=\infty\).

Lambda pilot:
Run reduced pilot on:
- all 3 rules
- first split only
- first 3 refs only
- radii `[0.05,0.10,0.20,0.45,0.80]`
- lambda candidates `[1,10,50,100]`

Select a single lambda for all rules using:
- q05 ESS fraction >= 0.02
- split logZ/P <= 0.004
- visible but not saturated phi_energy separation
- no rule-specific lambda

Full smoke radii:
`[0.05,0.10,0.20,0.45,0.80,1.20,1.60,2.00]`

Samples per ref/radius:
- 0.05: 256
- 0.10: 256
- 0.20: 512
- 0.45: 512
- 0.80: 1024
- 1.20: 2048
- 1.60: 2048
- 2.00: 4096

Implement:
- vMF sampling in dimension P.
- stable log_sphere_mgf / logM.
- split samples for split-logZ.
- logsumexp everywhere.
- unit summaries per ref/radius.
- summary-only mode available.

Outputs:
`02_dnn/08_mnist/runs/smoke/05_pool2_pm_sais_sampling/`
- `selected_lambda.json`
- `lambda_selection_report.csv`
- `shell_summary_by_unit.csv`
- `shell_summary_by_rule_radius.csv`
- `qc_by_rule_radius.csv`
- `figures/fig01_lambda_pilot_phi_energy.png`
- `figures/fig02_sampling_qc_ess_heatmap.png`
- `figures/fig03_sampling_qc_split_logz_heatmap.png`
- `figures/fig05_weighted_ce_by_rule_radius.png`
- `REPORT.md`
- `run_config_resolved.json`
- `QC_STATUS.json`

Tests:
Create and run:
`02_dnn/08_mnist/tests/test_stage05_pm_sais_math.py`

QC:
- hard shell distance correct.
- vMF samples unit norm.
- kappa formula correct.
- logZ finite.
- no-claim failed radii.
- bootstrap sd phi <= 0.012 for claimed radii.

Stop after Stage 05.
