# High-beta small-d energy CI comparison

- datasets: 30, 60, 90
- high beta: 0.29, 0.31, 0.33, 0.35, 0.37, 0.39
- small-d windows: d <= 0.10, d <= 0.30
- phi_energy CI source: final high_beta_curve_comparison tables
- dphi_energy_dr CI source: 30/60 dataset SEM comparison table plus 90 final unit-summary dataset aggregation
- d2phi_energy_dr2 CI source: finite difference of dphi_energy_dr with propagated SEM

## Figures

- `02_dnn/05_proxy_local_entropy/figures/high_beta_energy_derivatives_ci_30_60_90/energy_phi_d1_d2_ci_small_d_combined.png`
- `02_dnn/05_proxy_local_entropy/figures/high_beta_energy_derivatives_ci_30_60_90/phi_energy_ci_small_d.png`
- `02_dnn/05_proxy_local_entropy/figures/high_beta_energy_derivatives_ci_30_60_90/dphi_energy_dr_ci_small_d.png`
- `02_dnn/05_proxy_local_entropy/figures/high_beta_energy_derivatives_ci_30_60_90/d2phi_energy_dr2_ci_small_d.png`
- `02_dnn/05_proxy_local_entropy/figures/high_beta_energy_derivatives_ci_30_60_90/energy_phi_d1_d2_ci_tiny_d_combined.png`
- `02_dnn/05_proxy_local_entropy/figures/high_beta_energy_derivatives_ci_30_60_90/phi_energy_ci_tiny_d.png`
- `02_dnn/05_proxy_local_entropy/figures/high_beta_energy_derivatives_ci_30_60_90/dphi_energy_dr_ci_tiny_d.png`
- `02_dnn/05_proxy_local_entropy/figures/high_beta_energy_derivatives_ci_30_60_90/d2phi_energy_dr2_ci_tiny_d.png`

## Output tables

- `02_dnn/05_proxy_local_entropy/figures/high_beta_energy_derivatives_ci_30_60_90/energy_phi_d1_d2_ci_detail.csv`
- `02_dnn/05_proxy_local_entropy/figures/high_beta_energy_derivatives_ci_30_60_90/energy_phi_d1_d2_ci_width_summary.csv`

## Mean 95% CI half-width by beta and window

| metric | beta | window | 30 dataset | 60 dataset | 90 dataset |
|---|---:|---|---:|---:|---:|
| phi_energy | 0.29 | d_0p01_to_0p10 | 6.16889e-05 | 4.42559e-05 | 3.31725e-05 |
| phi_energy | 0.29 | d_0p01_to_0p30 | 6.93238e-05 | 4.89801e-05 | 3.68286e-05 |
| phi_energy | 0.31 | d_0p01_to_0p10 | 5.45251e-05 | 3.41843e-05 | 2.44493e-05 |
| phi_energy | 0.31 | d_0p01_to_0p30 | 6.246e-05 | 3.91298e-05 | 2.79976e-05 |
| phi_energy | 0.33 | d_0p01_to_0p10 | 6.10736e-05 | 3.54893e-05 | 2.56515e-05 |
| phi_energy | 0.33 | d_0p01_to_0p30 | 6.69642e-05 | 3.89825e-05 | 2.86284e-05 |
| phi_energy | 0.35 | d_0p01_to_0p10 | 5.33536e-05 | 3.36156e-05 | 2.42035e-05 |
| phi_energy | 0.35 | d_0p01_to_0p30 | 6.00325e-05 | 3.81133e-05 | 2.76075e-05 |
| phi_energy | 0.37 | d_0p01_to_0p10 | 4.5344e-05 | 2.87319e-05 | 2.07198e-05 |
| phi_energy | 0.37 | d_0p01_to_0p30 | 5.20879e-05 | 3.26083e-05 | 2.36615e-05 |
| phi_energy | 0.39 | d_0p01_to_0p10 | 4.55257e-05 | 2.72474e-05 | 2.02642e-05 |
| phi_energy | 0.39 | d_0p01_to_0p30 | 5.07184e-05 | 3.0625e-05 | 2.29149e-05 |
| dphi_energy_dr | 0.29 | d_0p01_to_0p10 | 6.95993e-05 | 4.25475e-05 | 3.19706e-05 |
| dphi_energy_dr | 0.29 | d_0p01_to_0p30 | 9.49627e-05 | 5.85247e-05 | 4.45292e-05 |
| dphi_energy_dr | 0.31 | d_0p01_to_0p10 | 6.92704e-05 | 4.15262e-05 | 2.99985e-05 |
| dphi_energy_dr | 0.31 | d_0p01_to_0p30 | 0.000116582 | 7.01641e-05 | 5.1068e-05 |
| dphi_energy_dr | 0.33 | d_0p01_to_0p10 | 5.77398e-05 | 3.35175e-05 | 3.38442e-05 |
| dphi_energy_dr | 0.33 | d_0p01_to_0p30 | 0.000100937 | 5.94347e-05 | 5.62686e-05 |
| dphi_energy_dr | 0.35 | d_0p01_to_0p10 | 6.15429e-05 | 3.95378e-05 | 3.0659e-05 |
| dphi_energy_dr | 0.35 | d_0p01_to_0p30 | 0.000106572 | 7.19073e-05 | 5.81778e-05 |
| dphi_energy_dr | 0.37 | d_0p01_to_0p10 | 5.90459e-05 | 3.5541e-05 | 2.9173e-05 |
| dphi_energy_dr | 0.37 | d_0p01_to_0p30 | 0.000114795 | 7.18358e-05 | 5.62137e-05 |
| dphi_energy_dr | 0.39 | d_0p01_to_0p10 | 6.49009e-05 | 3.64325e-05 | 2.67863e-05 |
| dphi_energy_dr | 0.39 | d_0p01_to_0p30 | 0.000119948 | 6.8549e-05 | 5.09571e-05 |
| d2phi_energy_dr2 | 0.29 | d_0p01_to_0p10 | 0.00519874 | 0.00317856 | 0.00238679 |
| d2phi_energy_dr2 | 0.29 | d_0p01_to_0p30 | 0.0070582 | 0.00435335 | 0.00331347 |
| d2phi_energy_dr2 | 0.31 | d_0p01_to_0p10 | 0.00515326 | 0.00309065 | 0.00223239 |
| d2phi_energy_dr2 | 0.31 | d_0p01_to_0p30 | 0.0086823 | 0.00522896 | 0.00380688 |
| d2phi_energy_dr2 | 0.33 | d_0p01_to_0p10 | 0.00433451 | 0.00250907 | 0.00252619 |
| d2phi_energy_dr2 | 0.33 | d_0p01_to_0p30 | 0.00754463 | 0.00444299 | 0.00419764 |
| d2phi_energy_dr2 | 0.35 | d_0p01_to_0p10 | 0.00458381 | 0.00294498 | 0.00228281 |
| d2phi_energy_dr2 | 0.35 | d_0p01_to_0p30 | 0.00796487 | 0.00537 | 0.00434472 |
| d2phi_energy_dr2 | 0.37 | d_0p01_to_0p10 | 0.00439326 | 0.0026447 | 0.00216977 |
| d2phi_energy_dr2 | 0.37 | d_0p01_to_0p30 | 0.00857839 | 0.00536666 | 0.00419582 |
| d2phi_energy_dr2 | 0.39 | d_0p01_to_0p10 | 0.00483344 | 0.00271158 | 0.00199372 |
| d2phi_energy_dr2 | 0.39 | d_0p01_to_0p30 | 0.0089476 | 0.0051166 | 0.00380393 |
