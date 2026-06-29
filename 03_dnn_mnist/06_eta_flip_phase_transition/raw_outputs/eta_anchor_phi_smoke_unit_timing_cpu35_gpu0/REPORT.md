# Eta Anchor Phi Smoke

- Status: `complete`
- Units: `1` / `1`
- Samples per unit: `128`
- Mean unit elapsed seconds: `2.795`
- Max unit elapsed seconds: `2.795`

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
| 0 | 1 | 1 | -0.0747031 | 0.285559 | 2.79468 |
