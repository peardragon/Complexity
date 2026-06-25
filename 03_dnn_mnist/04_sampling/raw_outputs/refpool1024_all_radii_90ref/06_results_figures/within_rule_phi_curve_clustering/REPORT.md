# Within-Rule Phi Curve Clustering

Generated at: 2026-06-18T13:24:50

Features: reference-level `delta_phi_energy_unit(d)` at all 25 radii. Numerical derivative features are not used in this run.

## Selected k

| rule | selected k | silhouette |
| --- | ---: | ---: |
| low_tv_spectral_teacher | 2 | 0.1063 |
| random_label | 2 | 0.3741 |
| real_even_odd | 2 | 0.0927 |
| teacher_nn | 2 | 0.1006 |

## Outputs

- `tables/phi_curve_features_by_ref.csv`
- `tables/within_rule_phi_curve_cluster_assignments.csv`
- `tables/within_rule_phi_curve_cluster_scores.csv`
- `tables/within_rule_phi_curve_cluster_summary.csv`
- `figures/fig01_within_rule_phi_curve_clusters.png`
- `figures/fig02_within_rule_phi_curve_cluster_pca.png`
