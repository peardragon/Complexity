# MNIST Refpool 1024 Mechanical Sampling

- Status: `complete`
- Units: `17640` / `17640`
- References per rule: `90`
- Radius grid: `advanced_mechanical_0p1_to_2p5_step0p05`
- Radii: `0.1..2.5` (49 values)
- Samples per unit: `1024`
- QC diagnostic pass rows: `8` / `196`

| rule | complete_radii | observed_units | missing_units | qc_diagnostic_pass_radii |
| --- | --- | --- | --- | --- |
| low_tv_spectral_teacher | 49 | 4410 | 0 | 3 |
| random_label | 49 | 4410 | 0 | 0 |
| real_even_odd | 49 | 4410 | 0 | 3 |
| teacher_nn | 49 | 4410 | 0 | 2 |

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
