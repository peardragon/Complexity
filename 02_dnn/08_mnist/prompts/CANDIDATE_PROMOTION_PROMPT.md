# CANDIDATE_PROMOTION_PROMPT

GOAL: Promote the MNIST14 PM-SAIS smoke run to production-candidate scale only if smoke QC passed.

Read:

- `02_dnn/08_mnist/runs/smoke/final_report/REPORT.md`
- `02_dnn/08_mnist/runs/smoke/final_report/final_claim_table.csv`
- `02_dnn/08_mnist/stages/07_candidate_final_promotion/README.md`

Candidate identity:

```text
mnist14_3rule_1024_5split_10ref
```

Scale:

- splits: 5
- train size: 1024
- test size: 2048
- references per dataset: 10
- common lambda_reg: selected by smoke
- estimator: PM-SAIS \(H=\infty\), optional \(H\in\{8,4,2\}\) diagnostic
- radius grid: smoke-supported grid, plus 2.80 stress only if smoke supported 2.00

Do not write to final `raw_outputs`; use `runs/candidate`.
