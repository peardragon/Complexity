# MNIST Refpool 1024 Mechanical Sampling

- Status: `partial`
- Units: `4` / `12`
- References per rule: `1`
- Radius grid: `custom_3_radii`
- Radii: `0.1..2.45` (3 values)
- Samples per unit: `1024`
- QC diagnostic pass rows: `2` / `12`

| rule | complete_radii | observed_units | missing_units | qc_diagnostic_pass_radii |
| --- | --- | --- | --- | --- |
| random_label | 0 | 0 | 3 | 0 |
| real_even_odd | 0 | 0 | 3 | 0 |
| teacher_nn | 0 | 0 | 3 | 0 |
| very_low_tv_spectral_teacher | 3 | 3 | 0 | 2 |

QC diagnostics are reported but are not used to select or skip sampling units.

Primary files:

- `04_reference_pool/reference_pool_index.csv`
- `05_pool2_pm_sais_sampling/shell_summary_by_unit.csv`
- `05_pool2_pm_sais_sampling/shell_summary_by_unit_with_phi.csv`
- `05_pool2_pm_sais_sampling/shell_summary_by_unit_with_phi_derivatives.csv`
- `05_pool2_pm_sais_sampling/shell_summary_by_rule_radius.csv`
- `05_pool2_pm_sais_sampling/qc_diagnostics_by_rule_radius.csv`
- `06_results_figures/phi_by_rule_radius.csv`
- `06_results_figures/dphi_dd_by_rule_radius.csv`
- `SAMPLING_STATUS.json`
