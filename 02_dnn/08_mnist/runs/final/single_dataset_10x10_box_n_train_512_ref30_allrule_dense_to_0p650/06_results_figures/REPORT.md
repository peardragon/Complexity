# MNIST10 Ref30 All-Rule Dense To 0.65

Selector: `dense_qc_stable_ref30`

Radii: `0.010, 0.011, 0.012, 0.013, 0.014, 0.016, 0.018, 0.020, 0.025, 0.030, 0.040, 0.050, 0.065, 0.080, 0.120, 0.150, 0.200, 0.300, 0.450, 0.650`

Rules: `low_tv_spectral_teacher, real_even_odd, teacher_nn, random_label`

## Current Decision

- Common QC-pass radii across all rules: `0.0100, 0.0110, 0.0120, 0.0130, 0.0140, 0.0160, 0.0180, 0.0200, 0.0250, 0.0300, 0.0400, 0.0500, 0.0650, 0.0800`
- Missing selected units: `534`
- Failed or incomplete selector/radius rows: `18`
- Split gate: `0.004`
- ESS gate: `0.04`
- Bootstrap SD gate: `0.012`

## Outputs

| file | rows |
| --- | ---: |
| overlay_unit_summary_long.csv | 3660 |
| selector_qc_by_rule_radius.csv | 80 |
| selector_phi_by_rule_radius.csv | 80 |
| missing_selector_units.csv | 534 |

## Figures

- `figures/fig01_phi_energy_qc_pass_allrule_dense_to_0p65.png`
- `figures/fig02_phi_energy_allrule_dense_to_0p65_diagnostic_panels.png`
- `figures/fig03_qc_pass_heatmap_dense_to_0p65.png`

## First Failed Or Incomplete Rows

| rule | radius | observed_ref_count | missing_ref_count | max_split_logZ_per_P_diff | bootstrap_sd_phi | claim_status |
| --- | --- | --- | --- | --- | --- | --- |
| random_label | 0.12 | 0 | 30 | nan | nan | missing_units |
| random_label | 0.15 | 0 | 30 | nan | nan | missing_units |
| random_label | 0.2 | 0 | 30 | nan | nan | missing_units |
| random_label | 0.3 | 0 | 30 | nan | nan | missing_units |
| random_label | 0.45 | 0 | 30 | nan | nan | missing_units |
| random_label | 0.65 | 0 | 30 | nan | nan | missing_units |
| real_even_odd | 0.12 | 0 | 30 | nan | nan | missing_units |
| real_even_odd | 0.15 | 0 | 30 | nan | nan | missing_units |
| real_even_odd | 0.2 | 0 | 30 | nan | nan | missing_units |
| real_even_odd | 0.3 | 0 | 30 | nan | nan | missing_units |
| real_even_odd | 0.45 | 0 | 30 | nan | nan | missing_units |
| real_even_odd | 0.65 | 0 | 30 | nan | nan | missing_units |
| teacher_nn | 0.12 | 1 | 29 | 0.0002151895759866181 | 0.0 | missing_units |
| teacher_nn | 0.15 | 1 | 29 | 0.000567478244629846 | 0.0 | missing_units |
| teacher_nn | 0.2 | 1 | 29 | 0.0019082769385028431 | 0.0 | missing_units |
| teacher_nn | 0.3 | 1 | 29 | 0.002596944907216163 | 0.0 | missing_units |
| teacher_nn | 0.45 | 1 | 29 | 0.0007701105651306512 | 0.0 | missing_units |
| teacher_nn | 0.65 | 1 | 29 | 0.0011966113864005266 | 0.0 | missing_units |
