# Eta Anchor Phi Smoke

- Status: `complete`
- Units: `20` / `20`
- Samples per unit: `128`
- Mean unit elapsed seconds: `4.408`
- Max unit elapsed seconds: `7.516`

This is not an eta-specific reference-search result. It reuses real_even_odd references as fixed anchors,
then evaluates eta-flipped label datasets around those anchors to measure timing and rough phi(d) behavior.

Primary files:

- `01_dataset_gen/eta_dataset_manifest.csv`
- `04_reference_pool/anchor_reference_index.csv`
- `05_pool2_pm_sais_sampling/shell_summary_by_unit_with_phi.csv`
- `06_results_figures/eta_anchor_phi_by_eta_radius.csv`
- `06_results_figures/eta_anchor_dphi_dd_by_eta_radius.csv`
- `06_results_figures/fig01_eta_anchor_phi_energy.png`

Summary preview:

| eta | radius | n_units | phi_energy_raw_mean | weighted_ce_mean | elapsed_s_mean |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.1 | 1 | -0.0033437 | 0.0103452 | 1.05912 |
| 0 | 0.8 | 1 | -0.0682563 | 0.249107 | 4.72618 |
| 0 | 0.9 | 1 | -0.0790375 | 0.276886 | 4.74726 |
| 0 | 1 | 1 | -0.0747031 | 0.285559 | 5.19061 |
| 0 | 1.1 | 1 | -0.0934629 | 0.34347 | 5.32808 |
| 0.2 | 0.1 | 1 | -0.221915 | 0.968282 | 3.9457 |
| 0.2 | 0.8 | 1 | -0.13151 | 0.551121 | 4.54756 |
| 0.2 | 0.9 | 1 | -0.131576 | 0.540373 | 4.49025 |
| 0.2 | 1 | 1 | -0.139215 | 0.595452 | 3.86561 |
| 0.2 | 1.1 | 1 | -0.142258 | 0.58712 | 3.91592 |
| 0.35 | 0.1 | 1 | -0.400984 | 1.6058 | 4.85019 |
| 0.35 | 0.8 | 1 | -0.156614 | 0.679523 | 3.59428 |
| 0.35 | 0.9 | 1 | -0.158151 | 0.700246 | 3.53518 |
| 0.35 | 1 | 1 | -0.162603 | 0.691043 | 3.52593 |
| 0.35 | 1.1 | 1 | -0.164347 | 0.698982 | 3.56365 |
| 0.5 | 0.1 | 1 | -0.454336 | 1.73015 | 7.51559 |
| 0.5 | 0.8 | 1 | -0.168065 | 0.709851 | 4.78425 |
| 0.5 | 0.9 | 1 | -0.162023 | 0.707396 | 4.68942 |
| 0.5 | 1 | 1 | -0.167258 | 0.695285 | 5.34312 |
| 0.5 | 1.1 | 1 | -0.162108 | 0.709321 | 4.94584 |
