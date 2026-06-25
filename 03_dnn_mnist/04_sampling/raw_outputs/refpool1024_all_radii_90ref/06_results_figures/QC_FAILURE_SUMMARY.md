# QC failure summary

## Thresholds

- split gate: `max_split_logZ_per_P_diff <= 0.004`
- ESS gate: `q05_ess_fraction >= 0.04`
- finite fraction gate: `finite_unit_fraction >= 0.95`
- bootstrap gate: `bootstrap_sd_delta_phi_energy <= 0.012`

## 60ref 1024-particle run

- units: `6000` complete rows
- unit finite failures: `0`
- unit ESS failures: `0`
- unit split failures: `2493`
- rule/radius QC pass rows: `6` / `100`

| rule | QC-pass radii | first fail radius | failing rule/radius rows | max split | max bootstrap sd | min q05 ESS |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| low_tv_spectral_teacher | 0.1, 0.2 | 0.3 | 23 | 0.030574 | 0.000855 | 0.556133 |
| random_label | none | 0.1 | 25 | 0.025517 | 0.001125 | 0.556376 |
| real_even_odd | 0.1, 0.2 | 0.3 | 23 | 0.028498 | 0.000846 | 0.548088 |
| teacher_nn | 0.1, 0.2 | 0.3 | 23 | 0.023693 | 0.000755 | 0.556142 |

Full table: `derived/qc_failure_breakdown_60ref.csv`

## 90ref 1024-particle run

- units: `9000` complete rows
- unit finite failures: `0`
- unit ESS failures: `0`
- unit split failures: `3734`
- rule/radius QC pass rows: `5` / `100`

| rule | QC-pass radii | first fail radius | failing rule/radius rows | max split | max bootstrap sd | min q05 ESS |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| low_tv_spectral_teacher | 0.1, 0.2 | 0.3 | 23 | 0.030574 | 0.000631 | 0.561437 |
| random_label | none | 0.1 | 25 | 0.025517 | 0.000864 | 0.565888 |
| real_even_odd | 0.1, 0.2 | 0.3 | 23 | 0.028498 | 0.000614 | 0.553703 |
| teacher_nn | 0.1 | 0.2 | 24 | 0.023693 | 0.000567 | 0.575100 |

Full table: `derived/qc_failure_breakdown_90ref.csv`

## Previous strict4096 diagnostic backup

This is not the same grid as the current 1024 run: it covers small/local-support radii `0.01...0.65`, not all `0.1...2.5` radii with 90 refs.

- baseline_4096: `1679` / `1725` units passed; `46` failed.
- replicate_fallback: `635` / `675` units passed; `40` failed.
