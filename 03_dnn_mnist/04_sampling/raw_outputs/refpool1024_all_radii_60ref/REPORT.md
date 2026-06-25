# MNIST Refpool 1024 Mechanical Sampling

- Status: `complete`
- Units: `6000` / `6000`
- References per rule: `60`
- Radii: `0.1..2.5 step 0.1` (25 values)
- Samples per unit: `1024`
- QC diagnostic pass rows: `6` / `100`

| rule | complete_radii | observed_units | missing_units | qc_diagnostic_pass_radii |
| --- | --- | --- | --- | --- |
| low_tv_spectral_teacher | 25 | 1500 | 0 | 2 |
| random_label | 25 | 1500 | 0 | 0 |
| real_even_odd | 25 | 1500 | 0 | 2 |
| teacher_nn | 25 | 1500 | 0 | 2 |

QC diagnostics are reported but are not used to select or skip sampling units.

Primary files:

- `04_reference_pool/reference_pool_index.csv`
- `05_pool2_pm_sais_sampling/shell_summary_by_unit.csv`
- `05_pool2_pm_sais_sampling/shell_summary_by_unit_with_phi.csv`
- `05_pool2_pm_sais_sampling/shell_summary_by_rule_radius.csv`
- `05_pool2_pm_sais_sampling/qc_diagnostics_by_rule_radius.csv`
- `06_results_figures/phi_by_rule_radius.csv`
- `SAMPLING_STATUS.json`
