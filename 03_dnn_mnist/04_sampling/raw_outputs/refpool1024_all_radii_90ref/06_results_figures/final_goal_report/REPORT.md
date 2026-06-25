# Final Goal Report: Refpool1024 Sampling Suitability And Raw Phi Clustering

Generated at: 2026-06-18T12:24:08

## Requirement Evidence

| requirement | evidence | status |
| --- | --- | --- |
| Additional experiment | expanded independent SMC replicate probe: 24 probe units x 2 replicates = 48 finite rows | done |
| Sampling suitability analysis | `stability_clustering/tables/sampling_suitability_by_rule_radius.csv` plus expanded probe tables | done |
| raw phi(d)_energy | `raw_phi_energy_by_rule_radius.csv`, `phi_energy = logZ_inf_full / P` | done |
| first derivative | `d_phi_energy_mean_curve_dd` in raw phi table | done |
| second derivative | `d2_phi_energy_mean_curve_dd2` in raw phi table | done |
| clustering using raw value features | global and within-rule reference cluster assignments/scores from raw phi + derivatives | done |

## Expanded Replicate Probe Summary

- Probe rows: `48`; finite rows: `48`.
- Split fail replicate rows: `16` / `48`.
- Probe design: 4 rules x 3 radii `(0.3, 1.0, 2.5)` x low/high source split refs x 2 independent seeds.

| rule | radius | source 90ref split fail rate | probe mean split fail rate | probe max phi sd | probe max split q95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| low_tv_spectral_teacher | 0.3 | 0.033 | 0.000 | 0.002023 | 0.003358 |
| low_tv_spectral_teacher | 1.0 | 0.356 | 0.750 | 0.001101 | 0.010655 |
| low_tv_spectral_teacher | 2.5 | 0.689 | 0.750 | 0.004565 | 0.013910 |
| random_label | 0.3 | 0.656 | 0.750 | 0.006194 | 0.008657 |
| random_label | 1.0 | 0.178 | 0.750 | 0.001133 | 0.010032 |
| random_label | 2.5 | 0.378 | 0.250 | 0.006801 | 0.010013 |
| real_even_odd | 0.3 | 0.033 | 0.000 | 0.000827 | 0.002345 |
| real_even_odd | 1.0 | 0.489 | 0.250 | 0.003401 | 0.005017 |
| real_even_odd | 2.5 | 0.556 | 0.250 | 0.003515 | 0.011446 |
| teacher_nn | 0.3 | 0.133 | 0.000 | 0.000629 | 0.002098 |
| teacher_nn | 1.0 | 0.444 | 0.250 | 0.003590 | 0.006833 |
| teacher_nn | 2.5 | 0.589 | 0.000 | 0.004792 | 0.002559 |

## Sampling Suitability From Existing 90ref Pool

| rule | q95 split pass radii | fail-rate<=5% radii | max split | max split fail rate | min ESS q05 |
| --- | ---: | ---: | ---: | ---: | ---: |
| low_tv_spectral_teacher | 3/25 | 3/25 | 0.030574 | 0.689 | 0.561437 |
| random_label | 0/25 | 0/25 | 0.025517 | 0.656 | 0.565888 |
| real_even_odd | 3/25 | 3/25 | 0.028498 | 0.678 | 0.553703 |
| teacher_nn | 2/25 | 2/25 | 0.023693 | 0.633 | 0.575100 |

## Raw Phi And Clustering

- Raw phi table rows: `100` rule/radius rows.
- Reference cluster assignment rows: `360` references.
- Global clustering selected `k=2, silhouette=0.4200` over k=2..8.
- Features used for clustering: raw `phi_energy`, `d_phi_energy_dd`, and `d2_phi_energy_dd2` over the 25 radii.

Global cluster composition:

| cluster | refs | rules | mean split fail rate |
| ---: | ---: | --- | ---: |
| 0 | 90 | random_label:90 | 0.328 |
| 1 | 270 | low_tv_spectral_teacher:90, real_even_odd:90, teacher_nn:90 | 0.444 |

Within-rule clustering:

| rule | selected k | silhouette |
| --- | ---: | ---: |
| low_tv_spectral_teacher | 2 | 0.0997 |
| random_label | 2 | 0.1034 |
| real_even_odd | 2 | 0.0644 |
| teacher_nn | 2 | 0.1070 |

## Figures

- `figures/fig01_expanded_probe_split_q95.png`
- `figures/fig02_expanded_probe_phi_energy_sd.png`
- `figures/fig03_source_vs_replicate_split_fail_rate.png`
- Existing raw phi derivative figure: `../stability_clustering/figures/fig03_raw_phi_energy_derivatives_by_rule.png`
- Existing clustering figure: `../stability_clustering/figures/fig04_global_reference_cluster_pca.png`

## Notes

- The saved `samples.npz` contains normalized target weights, so arbitrary random re-split logZ cannot be reconstructed from the saved pool alone. The expanded probe therefore uses fresh independent SMC seeds.
- `delta_phi` is not used for the final raw phi clustering features.
