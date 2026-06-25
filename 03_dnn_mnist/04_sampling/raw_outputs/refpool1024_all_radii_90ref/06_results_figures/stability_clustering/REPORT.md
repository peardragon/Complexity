# Sampling Suitability And Raw Phi Clustering

Generated at: 2026-06-18T11:56:50

## Scope

- Run: `refpool1024_all_radii_90ref`.
- Uses existing `n=1024` unit summaries and saved sample metadata.
- `phi(d)_energy` is the raw value `logZ_inf_full / P`, not delta from `d0`.
- The saved `samples.npz` contains normalized target weights, not per-particle unnormalized logZ contributions. Therefore arbitrary random re-split logZ estimates cannot be reconstructed from the saved pool alone; independent SMC replicates or richer saved SMC increments are required for true random multi-split logZ experiments.

## Sampling Suitability

| rule | q95 split pass radii | fail-rate<=5% radii | max split | max split fail rate | min ESS q05 |
| --- | ---: | ---: | ---: | ---: | ---: |
| low_tv_spectral_teacher | 3/25 | 3/25 | 0.030574 | 0.689 | 0.561437 |
| real_even_odd | 3/25 | 3/25 | 0.028498 | 0.678 | 0.553703 |
| teacher_nn | 2/25 | 2/25 | 0.023693 | 0.633 | 0.575100 |
| random_label | 0/25 | 0/25 | 0.025517 | 0.656 | 0.565888 |

## Raw Phi Energy

- Rule/radius rows: `100`.
- Derivatives are numerical derivatives on the radius grid using `numpy.gradient`.

## Clustering

- Global reference clustering selected `k=2, silhouette=0.4200` over k=2..8.
- Features: raw `phi_energy`, `d_phi_energy/dd`, and `d2_phi_energy/dd2` over all 25 radii.

Global cluster composition:

| cluster | refs | dominant rules | mean split fail rate |
| ---: | ---: | --- | ---: |
| 0 | 90 | random_label:90 | 0.328 |
| 1 | 270 | low_tv_spectral_teacher:90, real_even_odd:90, teacher_nn:90 | 0.444 |

Within-rule selected k:

| rule | k | silhouette |
| --- | ---: | ---: |
| low_tv_spectral_teacher | 2 | 0.0997 |
| random_label | 2 | 0.1034 |
| real_even_odd | 2 | 0.0644 |
| teacher_nn | 2 | 0.1070 |

## Outputs

- `tables/unit_sampling_suitability.csv`
- `tables/sampling_suitability_by_rule_radius.csv`
- `tables/raw_phi_energy_by_ref_radius.csv`
- `tables/raw_phi_energy_by_rule_radius.csv`
- `tables/reference_cluster_assignments_global.csv`
- `tables/reference_cluster_assignments_within_rule.csv`
- `figures/fig01_sampling_split_quantiles.png`
- `figures/fig02_sampling_split_fail_rate.png`
- `figures/fig03_raw_phi_energy_derivatives_by_rule.png`
- `figures/fig04_global_reference_cluster_pca.png`
- `figures/fig05_global_cluster_phi_energy_curves.png`
- `figures/fig06_within_rule_cluster_phi_energy_curves.png`
