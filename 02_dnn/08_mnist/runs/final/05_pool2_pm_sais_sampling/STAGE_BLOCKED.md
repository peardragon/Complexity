# STAGE_BLOCKED

Stage: `05_pool2_pm_sais_sampling`

## Exact Failing Condition

Partial final-scale Stage 05 PM-SAIS QC projection failed: existing 1024-particle unit summaries already exceed the candidate/final max split logZ/P threshold for many real_even_odd radii. Because the Stage 05 rule/radius gate uses the maximum split difference, adding more references with the same 1024-particle configuration cannot make those observed radii pass. Continuing the same full run would not produce the requested QC-supported 0.01..2.50 full curve.

## Observed
- unit_summaries_1024_or_larger: `4290`
- rule_radius_groups_observed: `250`
- rules_observed: `['real_even_odd']`
- real_even_odd_rule_radius_groups_observed: `250`
- real_even_odd_groups_already_over_split_threshold: `235`
- real_even_odd_groups_observed_split_pass: `15`
- split_threshold: `0.004`
- first_real_even_odd_failing_radius: `{'rule': 'real_even_odd', 'radius': 0.15, 'n_units_observed': 18, 'max_split_logZ_per_P_diff': 0.004585368152683314, 'q05_ess_fraction': 0.8732576140711519, 'split_gate_pass_observed': False}`
- worst_observed_rule_radius: `{'rule': 'real_even_odd', 'radius': 2.13, 'n_units_observed': 17, 'max_split_logZ_per_P_diff': 0.04977116950956985, 'q05_ess_fraction': 0.8193967994715761, 'split_gate_pass_observed': False}`

## Expected
- candidate_final_max_split_logZ_per_P_diff: `<= 0.004 per 02_dnn/08_mnist/04_QC_GATES.md`
- full_curve_requirement: `QC-supported radii across 0.01..2.50 for all three rules, or failed radii explicitly no_claim`

## Next Safe Action

Do not continue the same 1024-particle full Stage 05 run. First run an adaptive stability pilot on the failed radii/ref units using a stronger SMC configuration (for example higher target CESS, additional MH sweeps, and/or replicated independent SMC estimates) and only resume final sampling after the pilot demonstrates max split logZ/P <= 0.004 at representative mid/high radii for each rule.
