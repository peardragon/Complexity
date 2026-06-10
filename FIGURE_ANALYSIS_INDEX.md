# Figure Analysis Index

This file records which retained figures are intended for analysis after the
active-tree cleanup. The active DNN results are:

- `18_beta_cell_30_dataset_30_reference`
- `18_beta_cell_60_dataset_30_reference`

Figures not listed here should be treated as review-only or disposable unless a
stage README or run report says otherwise.

## Figure Policy

| Label | Meaning | Default handling |
| --- | --- | --- |
| Primary analysis | Directly supports the retained scientific claim or final interpretation. | Keep in active tree. |
| QC | Checks sampler, fit, reference geometry, or numerical quality. | Keep while reviewing; can be archived after acceptance. |
| Review | Human-facing inspection or interactive visualization. | Keep if useful for presentation; archive otherwise. |
| None retained | Stage is currently table/config driven. | Do not recreate figures unless needed for a report. |

## Stage Figure Table

| Stage | Active run / scope | Analysis figures | Role | What it is used for |
| --- | --- | --- | --- | --- |
| `01_theory/01_theory_analytic` | perceptron theory | None retained | None retained | Analytic values are consumed as tables by the comparison stage. |
| `01_theory/02_theory_sampling` | perceptron sampler | None retained | None retained | Sampling pool is consumed by the comparison stage. |
| `01_theory/03_theory_comparison` | perceptron L2 validation | `fig00_dense_qc_N_convergence_alpha0p1.png` | Primary analysis | Validates that the sampler converges toward the analytic L2-regularized perceptron shell-entropy reference. |
| `02_dnn/01_dataset_gen` | 18 beta x 30 and 18 beta x 60 datasets | None retained | None retained | Dataset evidence is currently `dataset_index.csv` plus raw NPZ/meta payloads. |
| `02_dnn/02_complexity_measure` | NMSTV complexity | None retained | None retained | Complexity evidence is currently the summary tables; regenerate figures only for presentation. |
| `02_dnn/03_reference_search` | 3NN references | None retained | None retained | Reference quality is table driven: `best_per_dataset.csv`, `best_by_cell.csv`, and coverage summaries. |
| `02_dnn/04_sampling` | `18_beta_cell_60_dataset_30_reference/d_0.01_to_2.50_dense` | `fig_sampling_q05_ess_fraction_by_radius.png`, `fig_sampling_split_logZ_per_P_by_radius.png` | QC | Main shell-sampler health checks: ESS and shell logZ behavior over radius. |
| `02_dnn/04_sampling` | `18_beta_cell_60_dataset_30_reference/d_0.01_to_2.50_dense` | `fig_smc_mh_acceptance_by_radius.png`, `fig_smc_min_cess_by_radius.png`, `fig_smc_step_count_by_radius.png` | QC | Adaptive SMC/MH diagnostics used to identify sampling instability or under-tempering. |
| `02_dnn/04_sampling` | `18_beta_cell_60_dataset_30_reference/d_0.01_to_2.50_dense` | `fig_weighted_accuracy_by_radius.png`, `fig_weighted_ce_l2_ratio_by_radius.png` | QC | Confirms sampled shell mass remains in the intended trained-solution regime. |
| `02_dnn/05_proxy_local_entropy` | combined 30/60 local-entropy analysis | `figures/local_entropy_dashboard.html` | Primary analysis | Interactive dashboard for both active runs. Use it to switch run, metric family, beta subset, radius window, linear/log radius scale, value transform, and line/heatmap view. |
| `02_dnn/05_proxy_local_entropy` | `18_beta_cell_30_dataset_30_reference/d_0.01_to_2.50_dense` | `local_entropy_dashboard.html` | Primary analysis | Single-run dashboard for the 30-dataset production result, generated from the retained summary tables with the same controls as the 60-dataset view. |
| `02_dnn/05_proxy_local_entropy` | `18_beta_cell_60_dataset_30_reference/d_0.01_to_2.50_dense` | `local_entropy_dashboard.html` | Primary analysis | Single-run dashboard for the 60-dataset extension, generated from the retained summary tables with the same controls as the 30-dataset view. |
| `02_dnn/06_reference_atlas` | `18_beta_cell_30_dataset_30_reference/d_0.01_to_2.50_dense` | `fig01_S_Q_H_vs_beta.png`, `fig02_S_H_phase_scatter.png` | Primary analysis | Reference-cloud geometry summary used to relate beta/complexity to reference-space structure. |
| `02_dnn/06_reference_atlas` | `18_beta_cell_30_dataset_30_reference/d_0.01_to_2.50_dense` | `fig03_pairwise_distance_cosine_heatmaps.png` | QC | Pairwise reference distance/cosine diagnostics. |
| `02_dnn/06_reference_atlas` | `18_beta_cell_30_dataset_30_reference/d_0.01_to_2.50_dense` | `fig04_linear_barrier_heatmap.png`, `fig05_Blin_vs_beta.png`, `fig06_reference_map_barrier_overlay.png` | Primary analysis | Straight-line barrier diagnostics showing how reference basins connect across beta. |
| `02_dnn/07_proxy_3D_landscape_for_visualize` | reviewed proxy surface | `fig01_phi_full_curve_matching_corrected.png`, `fig02_phi_curve_matching_residual_heatmap.png`, `fig03_fitted_results_target_vs_proxy.png` | QC | Validates that the fitted proxy surface matches target curves and reference-atlas descriptors. |
| `02_dnn/07_proxy_3D_landscape_for_visualize` | reviewed proxy surface | `fig05_proxy_contours_hq.png`, `fig06_proxy_3d_surfaces_hq.png`, `fig07_reference_centered_radial_profiles_fixed.png`, `fig08_radial_evidence_check.png` | Review | Human-facing visualization of the fitted local landscape. Useful for explanation, not the primary quantitative claim. |
| `02_dnn/07_proxy_3D_landscape_for_visualize` | browser artifacts | `proxy_landscape_v9_reviewed_site/index.html`, `proxy_landscape_v9_reviewed_standalone/proxy_3d_landscape.html` | Review | Interactive/standalone presentation artifacts for inspecting the proxy landscape. |

## Cleanup Notes

- DNN smoke-run figures were archived under the latest cleanup backup because
  they are not active final evidence.
- Active run reports and stage READMEs should point to this table when deciding
  whether a remaining figure is kept for analysis, QC, or presentation only.
