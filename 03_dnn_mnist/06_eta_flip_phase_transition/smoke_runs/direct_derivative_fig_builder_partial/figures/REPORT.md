# Direct Derivative Methodology Figures

This bundle combines the MNIST active-rule and eta-flip direct radial derivative runs.

Method contract:
- phi_E(d) is `logZ_inf_full / P`.
- The first derivative uses the stored radial-score estimator `dlogZ_inf_full_dr / P`.
- The second derivative is a finite difference of the stored first derivative along the radius grid.
- Rules and eta flips use the same shell SMC/resampling estimator after their own dataset/reference rows are loaded.

Run status:
- `rule_units`: `7335`
- `eta_units`: `9000`
- `total_units`: `16335`
- `groups`: `6`
- `radius_min`: `0.01`
- `radius_max`: `1.0`
- `p_dim`: `2461.0`

Figures:
- `fig01_rules_eta_phi_energy_spaghetti.png`
- `fig02_rules_eta_direct_dphi_dd_spaghetti.png`
- `fig03_rules_eta_direct_curvature_phase_like.png`
