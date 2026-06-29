# Phase Transition Discussion: 30ref Four-Eta MNIST

First-pass reproducible analysis for the current dense four-eta run. This report reads existing CSVs only; it does not rerun reference search, PM-SAIS sampling, or any expensive unit generation.

## Provenance

- Script: `/home/bjyong/Complexity/local_project/03_dnn_mnist/06_eta_flip_phase_transition/src/build_phase_transition_discussion_30ref.py`
- Eta run root: `/home/bjyong/Complexity/local_project/03_dnn_mnist/06_eta_flip_phase_transition/raw_outputs/eta_reference_phi_dense_4eta_30ref_r0p01_to_1p0_step0p01_n1024_cpu60_gpu0`
- Eta unit CSV: `/home/bjyong/Complexity/local_project/03_dnn_mnist/06_eta_flip_phase_transition/raw_outputs/eta_reference_phi_dense_4eta_30ref_r0p01_to_1p0_step0p01_n1024_cpu60_gpu0/05_pool2_pm_sais_sampling/shell_summary_by_unit_with_phi_derivatives.csv`
- Eta summary CSV: `/home/bjyong/Complexity/local_project/03_dnn_mnist/06_eta_flip_phase_transition/raw_outputs/eta_reference_phi_dense_4eta_30ref_r0p01_to_1p0_step0p01_n1024_cpu60_gpu0/06_results_figures/eta_reference_phi_by_eta_radius.csv`
- Even/odd rule run root: `/home/bjyong/Complexity/local_project/03_dnn_mnist/04_sampling/raw_outputs/active_rules_refpool1024_dense30ref_r0p01_to_1p0_step0p01_n1024_cpu60_gpu0`
- Even/odd unit CSV: `/home/bjyong/Complexity/local_project/03_dnn_mnist/04_sampling/raw_outputs/active_rules_refpool1024_dense30ref_r0p01_to_1p0_step0p01_n1024_cpu60_gpu0/05_pool2_pm_sais_sampling/shell_summary_by_unit_with_phi_derivatives.csv`
- Even/odd summary CSV: `/home/bjyong/Complexity/local_project/03_dnn_mnist/04_sampling/raw_outputs/active_rules_refpool1024_dense30ref_r0p01_to_1p0_step0p01_n1024_cpu60_gpu0/06_results_figures/phi_energy_by_rule_radius.csv`
- Eta NMSTV helper CSV: `/home/bjyong/Complexity/local_project/03_dnn_mnist/06_eta_flip_phase_transition/raw_outputs/eta_sweep_pilot_cpu35_gpu0/summary_by_eta_k.csv`

## Method

- Radius window: `0.01` to `1.0` on the shared 0.01-spaced grid.
- Baseline: `real_even_odd` from the dense 30-reference active-rule run, labelled `even odd` here.
- Eta cases: `eta=0.02, eta=0.05, eta=0.15, eta=0.25` from the final eta-specific-reference run.
- Derivative: per-reference finite difference of raw phi_E, then centered edge-padded moving-average smoothing with `7` radii.
- Curvature: finite difference of the smoothed derivative.
- A_kappa: radius integral of positive smoothed curvature, computed per reference and summarized by case.
- Gap metrics: group-mean eta curves minus the group-mean even/odd curve on the same radius grid. These are not reference-paired gaps.

## Case Summary

| case_label | n_refs | nmstv  | A_kappa_mean | A_kappa_sem | mean_curve_min_dphi_dr | mean_curve_min_dphi_dr_radius | phi_energy_raw_at_dmax |
| ---------- | ------ | ------ | ------------ | ----------- | ---------------------- | ----------------------------- | ---------------------- |
| eta=0.02   | 30     | 0.357  | 0.835        | 0.03202     | -0.1588                | 0.21                          | -0.0844                |
| eta=0.05   | 30     | 0.4306 | 0.9264       | 0.0328      | -0.2253                | 0.22                          | -0.09879               |
| eta=0.15   | 30     | 0.6551 | 1.145        | 0.02432     | -0.4352                | 0.21                          | -0.1344                |
| eta=0.25   | 30     | 0.8111 | 1.376        | 0.03251     | -0.567                 | 0.19                          | -0.1496                |
| even odd   | 30     | 0.4933 | 0.7509       | 0.02703     | -0.1278                | 0.2                           | -0.07414               |

## Gap Metrics To Even/Odd

| case_label | phi_rmse_gap | dphi_dr_rmse_gap | curvature_rmse_gap | A_kappa_gap_to_even_odd | A_kappa_ratio_to_even_odd |
| ---------- | ------------ | ---------------- | ------------------ | ----------------------- | ------------------------- |
| eta=0.02   | 0.008291     | 0.01828          | 0.6651             | 0.0841                  | 1.112                     |
| eta=0.05   | 0.02172      | 0.04493          | 0.7016             | 0.1755                  | 1.234                     |
| eta=0.15   | 0.05887      | 0.1307           | 1.342              | 0.3939                  | 1.525                     |
| eta=0.25   | 0.07872      | 0.1834           | 1.604              | 0.6252                  | 1.833                     |

## Outputs

- `/home/bjyong/Complexity/local_project/03_dnn_mnist/06_eta_flip_phase_transition/figures/phase_transition_discussion_30ref_eta0p02_0p05_0p15_0p25/curve_summary_by_case_radius.csv`
- `/home/bjyong/Complexity/local_project/03_dnn_mnist/06_eta_flip_phase_transition/figures/phase_transition_discussion_30ref_eta0p02_0p05_0p15_0p25/reflevel_derivative_curvature_by_case_radius.csv`
- `/home/bjyong/Complexity/local_project/03_dnn_mnist/06_eta_flip_phase_transition/figures/phase_transition_discussion_30ref_eta0p02_0p05_0p15_0p25/A_kappa_by_reference.csv`
- `/home/bjyong/Complexity/local_project/03_dnn_mnist/06_eta_flip_phase_transition/figures/phase_transition_discussion_30ref_eta0p02_0p05_0p15_0p25/case_summary_A_kappa.csv`
- `/home/bjyong/Complexity/local_project/03_dnn_mnist/06_eta_flip_phase_transition/figures/phase_transition_discussion_30ref_eta0p02_0p05_0p15_0p25/gap_to_even_odd_by_radius.csv`
- `/home/bjyong/Complexity/local_project/03_dnn_mnist/06_eta_flip_phase_transition/figures/phase_transition_discussion_30ref_eta0p02_0p05_0p15_0p25/gap_metrics_to_even_odd.csv`
- `/home/bjyong/Complexity/local_project/03_dnn_mnist/06_eta_flip_phase_transition/figures/phase_transition_discussion_30ref_eta0p02_0p05_0p15_0p25/fig01_raw_phi_E_even_odd_eta_comparison.png`
- `/home/bjyong/Complexity/local_project/03_dnn_mnist/06_eta_flip_phase_transition/figures/phase_transition_discussion_30ref_eta0p02_0p05_0p15_0p25/fig01_raw_phi_E_even_odd_eta_comparison.pdf`
- `/home/bjyong/Complexity/local_project/03_dnn_mnist/06_eta_flip_phase_transition/figures/phase_transition_discussion_30ref_eta0p02_0p05_0p15_0p25/fig02_smoothed_dphi_dr_even_odd_eta_comparison.png`
- `/home/bjyong/Complexity/local_project/03_dnn_mnist/06_eta_flip_phase_transition/figures/phase_transition_discussion_30ref_eta0p02_0p05_0p15_0p25/fig02_smoothed_dphi_dr_even_odd_eta_comparison.pdf`
- `/home/bjyong/Complexity/local_project/03_dnn_mnist/06_eta_flip_phase_transition/figures/phase_transition_discussion_30ref_eta0p02_0p05_0p15_0p25/fig03_curvature_A_kappa_even_odd_eta_comparison.png`
- `/home/bjyong/Complexity/local_project/03_dnn_mnist/06_eta_flip_phase_transition/figures/phase_transition_discussion_30ref_eta0p02_0p05_0p15_0p25/fig03_curvature_A_kappa_even_odd_eta_comparison.pdf`
- `/home/bjyong/Complexity/local_project/03_dnn_mnist/06_eta_flip_phase_transition/figures/phase_transition_discussion_30ref_eta0p02_0p05_0p15_0p25/fig04_gap_metrics_to_even_odd.png`
- `/home/bjyong/Complexity/local_project/03_dnn_mnist/06_eta_flip_phase_transition/figures/phase_transition_discussion_30ref_eta0p02_0p05_0p15_0p25/fig04_gap_metrics_to_even_odd.pdf`
- `/home/bjyong/Complexity/local_project/03_dnn_mnist/06_eta_flip_phase_transition/figures/phase_transition_discussion_30ref_eta0p02_0p05_0p15_0p25/run_config_resolved.json`
- `/home/bjyong/Complexity/local_project/03_dnn_mnist/06_eta_flip_phase_transition/figures/phase_transition_discussion_30ref_eta0p02_0p05_0p15_0p25/provenance_paths.json`
- `/home/bjyong/Complexity/local_project/03_dnn_mnist/06_eta_flip_phase_transition/figures/phase_transition_discussion_30ref_eta0p02_0p05_0p15_0p25/REPORT.md`

## Reproduction

```bash
python /home/bjyong/Complexity/local_project/03_dnn_mnist/06_eta_flip_phase_transition/src/build_phase_transition_discussion_30ref.py
```

The resolved inputs, parameters, and SHA-256 hashes are in `run_config_resolved.json` and `provenance_paths.json`.
