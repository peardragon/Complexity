GOAL: Execute Stage 07 only if Stage 06 smoke QC passed.

Read:
- `02_dnn/08_mnist/runs/smoke/final_report/REPORT.md`
- `02_dnn/08_mnist/runs/smoke/final_report/final_claim_table.csv`
- this stage README

Candidate:
- identity: `mnist14_3rule_1024_5split_10ref`
- splits: 5
- train size: 1024
- refs per dataset: 10
- use selected lambda from smoke
- use smoke-supported radii
- include 2.80 only as stress if smoke supports 2.00
- outputs under `02_dnn/08_mnist/runs/candidate/`

Final:
- identity: `mnist14_3rule_1024_10split_20ref`
- splits: 10
- train size: 1024
- refs per dataset: 20
- use candidate-selected settings
- outputs under `02_dnn/08_mnist/runs/final/`

Promotion rule:
- Do not promote if smoke/candidate fails QC.
- Do not change architecture/lambda per rule.
- Do not use final language for candidate.

Stop after writing candidate/final promotion report.
