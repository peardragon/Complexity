# STAGE_BLOCKED

Stage: `06_results_figures`

## Exact Failing Condition

Stage 06 refused to promote figures because the selected line is not all-QC-pass.

## Observed

- failed_rule_radius_rows: 5
- failed_rule_radius_rows_detail: [{'rule': 'random_label', 'radius': 0.02, 'q05_ess_fraction': 0.5604753686975648, 'max_split_logZ_per_P_diff': 0.0042931518535628925, 'bootstrap_sd_phi': 0.022801308427425424}, {'rule': 'random_label', 'radius': 0.03, 'q05_ess_fraction': 0.5883317661702359, 'max_split_logZ_per_P_diff': 0.004862503750040444, 'bootstrap_sd_phi': 0.015782337469124618}, {'rule': 'random_label', 'radius': 0.05, 'q05_ess_fraction': 0.58402878520018, 'max_split_logZ_per_P_diff': 0.0002820511262066597, 'bootstrap_sd_phi': 0.0207294507141033}, {'rule': 'random_label', 'radius': 0.08, 'q05_ess_fraction': 0.6930968810194097, 'max_split_logZ_per_P_diff': 0.002419989333714199, 'bootstrap_sd_phi': 0.019504508599686398}, {'rule': 'real_even_odd', 'radius': 0.05, 'q05_ess_fraction': 0.7753575054307802, 'max_split_logZ_per_P_diff': 0.0025063757281134033, 'bootstrap_sd_phi': 0.013665717074156484}]
- common_pass_radii: [0.01]

## Expected

- all_selected_rule_radius_qc_pass: True
- selected_radii: [0.01, 0.02, 0.03, 0.05, 0.08]

## Files Already Created

- none

## Next Safe Action

Return to Stage 05 and either narrow the selected radius line or run targeted stronger PM-SAIS for failed rows.
