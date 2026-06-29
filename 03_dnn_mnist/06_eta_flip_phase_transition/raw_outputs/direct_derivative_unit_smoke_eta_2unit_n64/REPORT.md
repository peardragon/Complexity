# Eta-Specific Reference Phi Smoke

- Status: `complete`
- Units: `2` / `2`
- Samples per unit: `64`
- Mean unit elapsed seconds: `0.720`
- Max split logZ/P diff: `0.00514848`
- Min ESS fraction: `0.625178`

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
0.02,eta_0p02,0.1,1,-0.004085377433217489,,0.0,0.0,,0.0,0.6251780945078831,0.6251780945078831,0.0005845893053386162,0.013288012146354009,0.003241616144604686,0.5545711517333984,-0.17590822803415693,,0.0,-0.17590822803415693,,0.0,-0.11159605315995369,,0.0,0.027945473935475404,0.0
0.02,eta_0p02,0.2,1,-0.02167620023663318,,0.0,-0.017590822803415694,,0.0,0.9850889693823148,0.9850889693823148,0.005148481170704752,0.07187532014271858,0.025488550786888782,0.8858354091644287,-0.17590822803415693,,0.0,-0.17590822803415693,,0.0,-0.19129736054191335,,0.0,0.06916717076939129,0.0

