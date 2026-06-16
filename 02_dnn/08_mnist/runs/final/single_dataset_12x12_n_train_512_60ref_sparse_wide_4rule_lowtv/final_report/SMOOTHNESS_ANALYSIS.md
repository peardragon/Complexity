# Smoothness Analysis: n_train=512 Sparse-Wide vs n_train=1024 Dense

This diagnostic compares the previous n_train=1024 dense raw curve with the new n_train=512 sparse-wide PM-SAIS run. The estimator and hard L2 shell methodology are unchanged; the new run changes train sample count, reference count, radius grid, and particles per rule/radius.

## Run Scale

| run | n_train | refs/rule | radii | samples/unit | shell units |
| --- | ---: | ---: | ---: | ---: | ---: |
| old dense | 1024 | 30 | 250 | mixed 256/1024 | 30000 |
| new sparse-wide | 512 | 60 | 19 | 2048 | 4560 |

## Energy Smoothness Summary

| rule | old_bootstrap_sd_median | new_bootstrap_sd_median | sd_ratio_new_over_old | old_second_diff_abs_median | new_second_diff_abs_median | roughness_ratio_new_over_old | new_qc_pass_radii |
| --- | --- | --- | --- | --- | --- | --- | --- |
| low_tv_spectral_teacher | 0.001640 | 0.000423 | 0.257876 | 0.002284 | 0.002089 | 0.914417 | 8 |
| random_label | 0.016928 | 0.000585 | 0.034544 | 0.015533 | 0.003191 | 0.205439 | 4 |
| real_even_odd | 0.001662 | 0.017904 | 10.775163 | 0.002350 | 0.005333 | 2.269228 | 1 |
| teacher_nn | 0.065295 | 0.000548 | 0.008398 | 0.005415 | 0.002191 | 0.404702 | 5 |

Interpretation: lower bootstrap SD and lower adjacent second-difference indicate a smoother diagnostic energy curve. The new run removes ESS collapse for random_label and teacher_nn and sharply lowers their bootstrap variability. real_even_odd still has high reference-level variability in the energy split, so its formal QC-pass coverage remains narrow even though the full phi curve is visually smooth.

## Figures

- `figures/fig10_old1024_vs_new512_phi_energy_comparison.png`
- `figures/fig11_new512_energy_and_full_phi_sparse_wide.png`