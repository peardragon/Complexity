# Eta-Specific Reference Phi Smoke

- Status: `complete`
- Units: `1` / `1`
- Samples per unit: `128`
- Mean unit elapsed seconds: `3.765`
- Max split logZ/P diff: `0.00241232`
- Min ESS fraction: `0.614015`

This smoke uses eta-specific exact references, unlike the earlier fixed real_even_odd anchor smoke.

Primary files:

- `04_reference_pool/reference_index.csv`
- `05_pool2_pm_sais_sampling/shell_summary_by_unit_with_phi.csv`
- `06_results_figures/eta_reference_phi_by_eta_radius.csv`
- `06_results_figures/eta_reference_dphi_dd_by_eta_radius.csv`
- `06_results_figures/fig01_eta_reference_phi_energy_d1_zoom.png`
- `06_results_figures/fig02_eta_reference_delta_phi_energy_d1_zoom.png`
- `06_results_figures/fig03_eta_reference_dphi_dd_d1_zoom.png`

Summary CSV preview:

eta,rule,radius,n_units,phi_energy_raw_mean,phi_energy_raw_sd,phi_energy_raw_sem,delta_phi_energy_mean,delta_phi_energy_sd,delta_phi_energy_sem,ess_fraction_min,ess_fraction_mean,split_logZ_per_P_diff_max,weighted_ce_mean,weighted_error_mean,elapsed_s_mean
0.35,eta_0p35,1.0,1,-0.16509158941951857,,0.0,,,0.0,0.614014917022083,0.614014917022083,0.002412322461254438,0.6838759533312357,0.38338671595951745,3.764542818069458

