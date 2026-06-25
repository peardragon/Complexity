# 02_dnn NMSTV vs Current Phi(d)

Stopped the CE/norm proxy comparison. This report uses the original 02_dnn Stage 02 complexity measure: graph TV/NMSTV over the dataset labels.

Inputs:

- Complexity: `/home/bjyong/Complexity/windows_project/02_dnn/08_mnist/runs/final/single_dataset_10x10_box_n_train_512_60ref_allrule_sparse_0p010_to_2p500/02_complexity_measure/complexity_by_rule_summary.csv`
- Current phi(d): `/home/bjyong/Complexity/local_project/03_dnn_mnist/04_sampling/raw_outputs/refpool1024_all_radii_90ref/06_results_figures/stability_clustering/tables/raw_phi_energy_by_rule_radius.csv`

Definition checked from 02_dnn Stage 02: kNN graph over standardized `X_train`, k in [8, 16, 32], weighted label TV normalized by `2p(1-p)` to produce NMSTV.

## Main Result

NMSTV order: low_tv < even_odd < teacher_nn < random

Current phi area magnitude order: even_odd < low_tv < teacher_nn < random

Current phi d=2.5 magnitude order: even_odd < low_tv < teacher_nn < random

| label | nmstv_mean | tv_mean | minus_phi_area | minus_phi_d0p65 | minus_phi_d1p0 | minus_phi_d2p5 | complexity_rank_nmstv_low_to_high | phi_area_rank_low_to_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| low_tv | 0.40706 | 0.20353 | 0.19685 | 0.06109 | 0.07738 | 0.12025 | 1 | 2 |
| even_odd | 0.49329 | 0.24664 | 0.19079 | 0.05872 | 0.07437 | 0.12004 | 2 | 1 |
| teacher_nn | 0.68438 | 0.34219 | 0.25763 | 0.08459 | 0.10467 | 0.15107 | 3 | 3 |
| random | 0.98556 | 0.49278 | 0.38194 | 0.15920 | 0.16079 | 0.17781 | 4 | 4 |

## Correlation, n=4 Rules Only

| phi_feature | pearson_r | pearson_p | spearman_r | spearman_p |
| --- | --- | --- | --- | --- |
| minus_phi_area | 0.9810 | 0.0190 | 0.8000 | 0.2000 |
| minus_phi_d0p65 | 0.9678 | 0.0322 | 0.8000 | 0.2000 |
| minus_phi_d1p0 | 0.9805 | 0.0195 | 0.8000 | 0.2000 |
| minus_phi_d2p5 | 0.9849 | 0.0151 | 0.8000 | 0.2000 |

## Interpretation

- Yes: the rule-based dataset complexity in 02_dnn is TV/NMSTV, not reference CE/norm.
- Current 90-ref phi(d) broadly follows NMSTV: random is highest, teacher_nn is next, and the two structured rules are lowest.
- The only mismatch is the bottom pair: NMSTV has low_tv below real_even_odd, while current phi magnitude is nearly tied and slightly puts low_tv above real_even_odd across several radii. The gap is small compared with teacher_nn/random separation.
- Therefore, for rule-level comparison, use NMSTV as the dataset complexity axis. CE/test/norm should be treated only as reference-search or trained-reference diagnostics.

## Files

- `tables/nmstv_phi_rule_summary.csv`
- `tables/nmstv_phi_curve_joined_by_rule_radius.csv`
- `tables/nmstv_phi_correlations_n4.csv`
- `tables/nmstv_phi_rank_comparison.csv`
- `figures/fig01_nmstv_order_and_current_phi_curves.png`
- `figures/fig02_nmstv_vs_phi_scatter.png`
- `figures/fig03_phi_rank_by_radius_against_nmstv_order.png`
