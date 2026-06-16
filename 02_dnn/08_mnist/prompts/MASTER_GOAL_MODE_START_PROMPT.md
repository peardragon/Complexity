# MASTER_GOAL_MODE_START_PROMPT

GOAL: Execute the full MNIST14 PM-SAIS smoke pipeline under `02_dnn/08_mnist`.

First read these files:

1. `02_dnn/08_mnist/README.md`
2. `02_dnn/08_mnist/AGENTS.md`
3. `02_dnn/08_mnist/00_GLOBAL_GOAL.md`
4. `02_dnn/08_mnist/01_FORMULAS_PM_SAIS.md`
5. every `02_dnn/08_mnist/stages/*/README.md`

Then execute stages 00 through 06 in order at smoke scale.

## Required stage order

1. `00_repo_audit`
2. `01_dataset_prepare`
3. `02_complexity_measure`
4. `03_pool_design`
5. `04_exact_reference_search`
6. `05_pool2_pm_sais_sampling`
7. `06_results_figures`

## Smoke scale

- splits: 3
- train size: 256
- test size: 2048
- label rules: `real_even_odd`, `teacher_nn`, `random_label`
- architecture: `196-16-16-1-tanh`, `P=3441`
- selected references per dataset: 5
- radius grid: `[0.05, 0.10, 0.20, 0.45, 0.80, 1.20, 1.60, 2.00]`
- main estimator: PM-SAIS \(H=\infty\)
- optional diagnostic: \(H\in\{8,4,2\}\)

## Strict constraints

- Work under `02_dnn/08_mnist`.
- Do not overwrite retained production outputs outside this directory.
- Use existing repo code only by reading, importing, or copying into the new namespace with backward-compatible behavior.
- Do not use generated images.
- Every figure must be generated from actual data.
- Stop on hard QC failure and write `STAGE_BLOCKED.md`.
- Failed radii are `no_claim`.
- Do not claim final production from smoke.

## Required final output

At the end of Stage 06, produce:

```text
02_dnn/08_mnist/runs/smoke/final_report/REPORT.md
02_dnn/08_mnist/runs/smoke/final_report/phi_by_rule_radius.csv
02_dnn/08_mnist/runs/smoke/final_report/qc_pass_by_rule_radius.csv
02_dnn/08_mnist/runs/smoke/final_report/final_claim_table.csv
02_dnn/08_mnist/runs/smoke/final_report/figures/fig04_phi_energy_three_rules_main.png
```

Print:

1. final supported d_raw radii;
2. no-claim d_raw radii;
3. main result sentence;
4. exact command or prompt for candidate promotion.
