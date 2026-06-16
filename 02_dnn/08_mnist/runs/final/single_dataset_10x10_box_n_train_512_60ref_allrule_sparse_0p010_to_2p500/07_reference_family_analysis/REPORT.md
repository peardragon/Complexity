# MNIST10 Reference Family Analysis

Run root: `02_dnn/08_mnist/runs/final/single_dataset_10x10_box_n_train_512_60ref_allrule_sparse_0p010_to_2p500`

This analysis decomposes the optimizer-induced exact reference ensemble into reference-level phi(d) and QC behavior. It is diagnostic unless a selector is explicitly marked predeclared.

## Main Finding

Large-domain sparse production is **not supported by the current evidence** for a predeclared 30-reference low_tv selector.

Decision: `False`.

Next safe action: Do not launch sparse large-domain production from the current evidence. First define a predeclared reference law such as l2_min_norm_ref30, run a targeted Stage05 pilot on its missing/hard large-radii units, and only promote if complete selector-level QC passes.

## Selector QC Summary

| selector | rule | qc-pass rows | max pass radius | missing rows | QC-fail rows |
| --- | --- | --- | --- | --- | --- |
| dense_qc_stable_ref30 | low_tv_spectral_teacher | 5 | 0.0800 | 14 | 10 |
| dense_qc_stable_ref30 | random_label | 5 | 0.0800 | 14 | 0 |
| dense_qc_stable_ref30 | real_even_odd | 5 | 0.0800 | 14 | 0 |
| dense_qc_stable_ref30 | teacher_nn | 5 | 0.0800 | 14 | 0 |
| high_margin_ref30 | low_tv_spectral_teacher | 5 | 0.0800 | 14 | 10 |
| high_margin_ref30 | random_label | 5 | 0.0800 | 14 | 0 |
| high_margin_ref30 | real_even_odd | 5 | 0.0800 | 14 | 0 |
| high_margin_ref30 | teacher_nn | 5 | 0.0800 | 14 | 0 |
| l2_min_norm_ref30 | low_tv_spectral_teacher | 5 | 0.0800 | 14 | 9 |
| l2_min_norm_ref30 | random_label | 5 | 0.0800 | 14 | 0 |
| l2_min_norm_ref30 | real_even_odd | 5 | 0.0800 | 14 | 0 |
| l2_min_norm_ref30 | teacher_nn | 5 | 0.0800 | 14 | 0 |
| optimizer_first30_ref30 | low_tv_spectral_teacher | 9 | 0.3000 | 3 | 10 |
| optimizer_first30_ref30 | random_label | 5 | 0.0800 | 14 | 0 |
| optimizer_first30_ref30 | real_even_odd | 5 | 0.0800 | 14 | 0 |
| optimizer_first30_ref30 | teacher_nn | 5 | 0.0800 | 14 | 0 |
| phi_qc_cluster_stable_ref30 | low_tv_spectral_teacher | 5 | 0.0800 | 14 | 2 |
| phi_qc_cluster_stable_ref30 | random_label | 5 | 0.0800 | 14 | 0 |
| phi_qc_cluster_stable_ref30 | real_even_odd | 5 | 0.0800 | 14 | 0 |
| phi_qc_cluster_stable_ref30 | teacher_nn | 5 | 0.0800 | 14 | 0 |
| qc_hardness_stable_ref30 | low_tv_spectral_teacher | 5 | 0.0800 | 14 | 0 |
| qc_hardness_stable_ref30 | random_label | 5 | 0.0800 | 14 | 0 |
| qc_hardness_stable_ref30 | real_even_odd | 5 | 0.0800 | 14 | 0 |
| qc_hardness_stable_ref30 | teacher_nn | 5 | 0.0800 | 14 | 0 |

## Hard low_tv Reference Examples

| ref | fail count | first fail d | max split | theta norm | min margin |
| --- | --- | --- | --- | --- | --- |
| 6 | 8 | 0.6500 | 0.014484 | 11.676 | 0.685 |
| 19 | 5 | 1.0000 | 0.019921 | 13.515 | 1.798 |
| 16 | 5 | 1.5000 | 0.012254 | 13.998 | 2.955 |
| 21 | 5 | 0.6500 | 0.011820 | 12.138 | 1.640 |
| 10 | 5 | 0.8500 | 0.010978 | 11.694 | 1.355 |
| 24 | 5 | 1.2500 | 0.010563 | 13.559 | 0.657 |
| 9 | 4 | 1.5000 | 0.018550 | 14.850 | 1.996 |
| 12 | 4 | 1.5000 | 0.013506 | 12.635 | 1.580 |

## Outputs

| artifact | path |
| --- | --- |
| reference diagnostics | reference_diagnostics.csv |
| selector membership | selector_membership.csv |
| selector QC | selector_qc_by_rule_radius.csv |
| selector phi | selector_phi_by_rule_radius.csv |
| large-domain decision | large_domain_decision.json |
| low_tv phi spaghetti | figures/fig01_lowtv_reference_phi_energy_spaghetti.png |
| low_tv split heatmap | figures/fig02_lowtv_split_logz_heatmap.png |
| low_tv selector phi | figures/fig03_lowtv_selector_phi_energy.png |
| low_tv selector QC map | figures/fig04_lowtv_selector_qc_pass_map.png |
| low_tv family scatter | figures/fig05_lowtv_reference_family_scatter.png |

## Interpretation

The current 60-reference low_tv ensemble behaves like a mixture of effective reference families. Several references repeatedly fail split-logZ stability despite acceptable ESS, so simply increasing sample count is not enough evidence for a large-domain all-reference claim. The optimizer-first, l2-min-norm, high-margin, and dense-QC-stable 30-reference selectors are predeclared diagnostics here; any production claim must rerun or complete their missing large-radius units and pass selector-level QC before promotion.
