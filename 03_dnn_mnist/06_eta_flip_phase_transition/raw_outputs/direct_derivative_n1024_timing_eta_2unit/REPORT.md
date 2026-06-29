# Eta-Specific Reference Phi Smoke

- Status: `complete`
- Units: `2` / `2`
- Samples per unit: `1024`
- Mean unit elapsed seconds: `14.512`
- Max split logZ/P diff: `0.00988664`
- Min ESS fraction: `0.670031`

This smoke uses eta-specific exact references, unlike the earlier fixed real_even_odd anchor smoke.

Primary files:

- `04_reference_pool/reference_index.csv`
- `05_pool2_pm_sais_sampling/shell_summary_by_unit_with_phi.csv`
- `05_pool2_pm_sais_sampling/shell_summary_by_unit_with_phi_derivatives.csv`
- `06_results_figures/eta_reference_phi_by_eta_radius.csv`
- `06_results_figures/eta_reference_dphi_dd_by_eta_radius.csv`
- `06_results_figures/fig01_eta_reference_phi_energy_d1_zoom.png`
- `06_results_figures/fig02_eta_reference_delta_phi_energy_d1_zoom.png`
- `06_results_figures/fig03_eta_reference_dphi_dd_d1_zoom.png`

Summary CSV preview:

eta,rule,radius,n_units,phi_energy_raw_mean,phi_energy_raw_sd,phi_energy_raw_sem,delta_phi_energy_mean,delta_phi_energy_sd,delta_phi_energy_sem,ess_fraction_min,ess_fraction_mean,split_logZ_per_P_diff_max,weighted_ce_mean,weighted_error_mean,elapsed_s_mean,d_phi_energy_raw_dd_unit_mean,d_phi_energy_raw_dd_unit_sd,d_phi_energy_raw_dd_unit_sem,d_delta_phi_energy_dd_unit_mean,d_delta_phi_energy_dd_unit_sd,d_delta_phi_energy_dd_unit_sem,d_phi_energy_direct_dd_unit_mean,d_phi_energy_direct_dd_unit_sd,d_phi_energy_direct_dd_unit_sem,split_dlogZ_dr_per_P_diff_max,ce_replay_max_abs_diff_max
0.02,eta_0p02,0.1,1,-0.004281770893815848,,0.0,0.0,,0.0,0.6700310665165052,0.6700310665165052,0.0003047229743290203,0.012987068639940421,0.003269618105249324,6.235193490982056,-0.08050627810157865,,0.0,-0.08050627810157865,,0.0,-0.10623809514191632,,0.0,0.0022982672325999754,0.0
0.02,eta_0p02,1.0,1,-0.07673742118523663,,0.0,-0.07245565029142079,,0.0,0.9999208830387105,0.9999208830387105,0.009886637484234396,0.29817066885251703,0.12163232846815455,22.789199829101562,-0.08050627810157865,,0.0,-0.08050627810157865,,0.0,-0.032790157261973756,,0.0,0.008266023185101655,0.0

