# MNIST Refpool 1024 Mechanical Sampling

- Status: `complete`
- Units: `12000` / `12000`
- References per rule: `30`
- Radius grid: `custom_100_radii`
- Radii: `0.01..1` (100 values)
- Samples per unit: `1024`
- QC diagnostic pass rows: `94` / `400`

| rule | complete_radii | observed_units | missing_units | qc_diagnostic_pass_radii |
| --- | --- | --- | --- | --- |
| random_label | 100 | 3000 | 0 | 8 |
| real_even_odd | 100 | 3000 | 0 | 28 |
| teacher_nn | 100 | 3000 | 0 | 19 |
| very_low_tv_spectral_teacher | 100 | 3000 | 0 | 39 |

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
