# FINAL_PROMOTION_PROMPT

GOAL: Run paper-level MNIST14 PM-SAIS final only after candidate QC passes.

Final identity:

```text
mnist14_3rule_1024_10split_20ref
```

Scale:

- 3 label rules
- 10 splits
- 20 references per dataset
- 1024 train examples per split
- common architecture and lambda selected by candidate
- main estimator: PM-SAIS \(H=\infty\)
- optional H-ladder diagnostic: \(H\in\{8,4,2\}\)

Do not change hyperparameters per rule.

Required final files:

```text
FINAL_REPORT.md
final_phi_by_rule_radius.csv
final_qc_by_rule_radius.csv
final_reference_summary.csv
final_complexity_summary.csv
final_claim_table.csv
fig04_phi_energy_three_rules_main.png
fig05_phi_full_three_rules.png
fig07_final_qc_pass_heatmap.png
fig11_final_storyboard.png
```

Final report must explicitly say:

- exact supported d_raw range;
- no-claim radii;
- reference ensemble is optimizer-induced;
- full phi includes shell-area dominance;
- phi_energy is the main landscape-quality comparison;
- random-label test accuracy is not generalization.
