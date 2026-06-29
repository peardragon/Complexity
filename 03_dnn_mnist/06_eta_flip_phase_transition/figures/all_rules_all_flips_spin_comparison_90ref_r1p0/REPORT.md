# All Rules + All 90ref Flips vs Previous Spin 3NN: Data-State Audit

## Scope

- MNIST 4-rule advanced: 4 rules, 90 references, `n=1024`, radius `0.10..2.50` step `0.05`; figures here use the common `0.10..1.00` window.
- MNIST flip advanced: all flip cases with 90-reference phi sampling currently available: eta `0.25,0.30,0.35,0.40`, radius `0.10..1.00` step `0.05`, `n=1024`.
- Dense eta/ref1 flip outputs exist, but are not mixed into the main 90ref figure because their reference count and radius support are not comparable.
- Previous spin 3NN comparison uses `02_dnn/05_proxy_local_entropy/.../18_beta_cell_90_dataset_30_reference/d_0.01_to_2.50_dense` and the spin-only positive-curvature-mass figure/table.

## Key Findings

- In one common plot, the flip eta curves sit far below the MNIST rule curves in raw `phi_E(d)` over `d<=1.0`; the separation is much larger than the rule-to-rule spread.
- MNIST rule/flip first derivatives in these retained outputs are posthoc finite differences along radius; they are not sampler-stored `dlogZ/dd` values.
- Previous spin 3NN stores a direct first derivative column, `mean_dlogZ_inf_full_dr`, and `dphi_energy_dr`; its second-derivative/curvature-style analyses are downstream finite differences from that stored first derivative.
- Spin has much denser radius support (`0.01` spacing, 250 radii, `0.01..2.50`) and many more aggregate references per beta/radius (`2700`) than current MNIST flip (`19` radii, 90 refs).
- Therefore, phase-like curvature mass is qualitatively comparable but not precision-matched: MNIST flip `A_kappa` is based on finite-difference first derivatives and a coarser `0.05` radius grid.

## Data State Table

| system                           | cases                | case_values                                                                               | unit_count | expected_unit_count | reference_count_basis                           | samples_per_ref_radius                                       | save_unit_samples_npz                   | derivative_status                                                                                     | analysis_basis                                                          | radius_count | radius_min | radius_max | radius_step_min | radius_step_max | radius_unique_steps | source_path                                                                                                                                                             |
| -------------------------------- | -------------------- | ----------------------------------------------------------------------------------------- | ---------- | ------------------- | ----------------------------------------------- | ------------------------------------------------------------ | --------------------------------------- | ----------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- | ------------ | ---------- | ---------- | --------------- | --------------- | ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| MNIST 4-rule advanced            | 4 rules              | very_low_tv_spectral_teacher,real_even_odd,teacher_nn,random_label                        | 17640      | 17640               | 90 refs per rule/radius                         | 1024                                                         | True                                    | posthoc finite difference from saved phi(d), not sampler-stored dlogZ/dd                              | diagnostic; QC not used to skip sampling                                | 49           | 0.1        | 2.5        | 0.05            | 0.05            | 0.05                | /home/bjyong/Complexity/local_project/03_dnn_mnist/04_sampling/raw_outputs/very_low_tv_spectral_teacher_refpool1024_advanced_90ref                                      |
| MNIST label-flip advanced        | 4 eta flips at 90ref | 0.25,0.3,0.35,0.4                                                                         | 6840       | 6840                | 90 refs per eta/radius                          | 1024                                                         | False                                   | posthoc finite difference from saved phi(d), not sampler-stored dlogZ/dd                              | complete for d<=1.0; split diagnostic max exceeds old strict 0.004 gate | 19           | 0.1        | 1.0        | 0.05            | 0.05            | 0.05                | /home/bjyong/Complexity/local_project/03_dnn_mnist/06_eta_flip_phase_transition/raw_outputs/eta_reference_phi_advanced_4eta_90ref_r0p1_to_1p0_step0p05_n1024_cpu60_gpu0 |
| MNIST label-flip dense eta smoke | 6 eta flips at ref1  | 0.2,0.25,0.3,0.35,0.4,0.5                                                                 | 24         | 24                  | 1 ref per eta/radius                            | 1024                                                         | False                                   | insufficient/ref1 exploratory; not comparable to 90ref curves                                         | orientation only                                                        | 4            | 0.8        | 1.1        | 0.1             | 0.1             | 0.1                 | /home/bjyong/Complexity/local_project/03_dnn_mnist/06_eta_flip_phase_transition/raw_outputs/eta_reference_phi_dense_eta_ref1_d1_n1024_cpu35_gpu0                        |
| previous 3NN spin                | 18 beta cells        | 0.05,0.07,0.09,0.11,0.13,0.15,0.17,0.19,0.21,0.23,0.25,0.27,0.29,0.31,0.33,0.35,0.37,0.39 | 4500       | 4500                | 2700 aggregate per beta/radius in summary table | stored in upstream shell pool, not explicit in proxy summary | not inspected in retained proxy summary | first derivative stored as mean_dlogZ_inf_full_dr / P; second derivative finite difference downstream | all proxy summary rows claim=pass                                       | 250          | 0.01       | 2.5        | 0.01            | 0.01            | 0.01                | /home/bjyong/Complexity/local_project/02_dnn/05_proxy_local_entropy/raw_outputs/18_beta_cell_90_dataset_30_reference/d_0.01_to_2.50_dense/summary_tables                |

## Available Flip Phi Runs

| run_name                                                                    | role                   | status   | completed_units | expected_units | eta_values                | radius_count | radius_min | radius_max | n_units_values |
| --------------------------------------------------------------------------- | ---------------------- | -------- | --------------- | -------------- | ------------------------- | ------------ | ---------- | ---------- | -------------- |
| eta_reference_phi_4eta_1ref_d1_n1024_cpu35_gpu0                             | support_1ref           | complete | 20              | 20             | 0,0.2,0.35,0.5            | 5            | 0.1        | 1.1        | 1              |
| eta_reference_phi_4eta_3ref_d1_n128_cpu35_gpu0                              | support_3ref_n128      | complete | 60              | 60             | 0,0.2,0.35,0.5            | 5            | 0.1        | 1.1        | 3              |
| eta_reference_phi_advanced_4eta_90ref_r0p1_to_1p0_step0p05_n1024_cpu60_gpu0 | main_90ref             | complete | 6840            | 6840           | 0.25,0.3,0.35,0.4         | 19           | 0.1        | 1.0        | 90             |
| eta_reference_phi_dense_eta_ref1_d1_n1024_cpu35_gpu0                        | support_dense_eta_ref1 | complete | 24              | 24             | 0.2,0.25,0.3,0.35,0.4,0.5 | 4            | 0.8        | 1.1        | 1              |
| eta_reference_phi_promoted_4eta_10ref_d1_n1024_cpu35_gpu0                   | support_10ref          | complete | 400             | 400            | 0.25,0.3,0.35,0.4         | 10           | 0.1        | 1.0        | 10             |
| eta_reference_phi_unit_timing_cpu35_gpu0                                    | timing_only            | complete | 1               | 1              | 0.35                      | 1            | 1.0        | 1.0        | 1              |

## Phi Range by MNIST Source

```text
    source       min       max
mnist_flip -0.159116 -0.019870
mnist_rule -0.160788 -0.003049
```

## MNIST Positive Curvature Mass by Case

```text
    source         case_label  n_refs  positive_curvature_mass_mean  positive_curvature_mass_sem
mnist_flip      flip eta=0.25      90                      0.674542                     0.007943
mnist_flip      flip eta=0.30      90                      0.675712                     0.008238
mnist_flip      flip eta=0.35      90                      0.741883                     0.007885
mnist_flip      flip eta=0.40      90                      0.816964                     0.008686
mnist_rule  rule: very low tv      90                      0.177100                     0.005918
mnist_rule     rule: even odd      90                      0.226506                     0.007153
mnist_rule   rule: teacher nn      90                      0.306818                     0.008864
mnist_rule rule: random label      90                      0.847440                     0.012080
```

## Previous Spin A_kappa

```text
 beta  A_kappa
 0.05 0.006388
 0.07 0.005908
 0.09 0.005199
 0.11 0.004639
 0.13 0.003830
 0.15 0.003064
 0.17 0.001880
 0.19 0.001003
 0.21 0.000257
 0.23 0.000018
 0.25 0.000000
 0.27 0.000000
 0.29 0.000000
 0.31 0.000000
 0.33 0.000000
 0.35 0.000029
 0.37 0.000004
 0.39 0.000013
```

## Generated Artifacts

- `fig01_all_rules_all_flips_phi_energy.png`: all 4 MNIST rules plus all 90ref flip eta curves on common `d<=1.0` axes.
- `fig02_all_rules_all_flips_derivative_curvature.png`: finite-difference first and second derivative summaries for MNIST groups.
- `fig03_mnist_flip_vs_spin_phase_metrics.png`: normalized curvature-mass/order-parameter comparison against previous spin 3NN.
- `fig04_all_available_flip_phi_runs_with_rules.png`: all phi-bearing flip runs overlaid with 4 rules; precision classes are visually separated.
- `data_state_comparison.csv`: grid/ref/sample/derivative-state audit table.
- `available_flip_phi_runs.csv`: inventory of every flip phi run currently found under `06_eta_flip_phase_transition/raw_outputs`.
- `all_mnist_phi_groups.csv`, `all_mnist_derivative_groups.csv`, `all_mnist_positive_curvature_mass.csv`: numeric tables backing the figures.
