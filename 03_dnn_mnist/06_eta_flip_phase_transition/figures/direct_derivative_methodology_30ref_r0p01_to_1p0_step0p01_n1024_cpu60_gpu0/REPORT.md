# Direct Derivative Methodology Figures

This bundle combines the MNIST active-rule and eta-flip direct radial derivative runs.

Method contract:
- phi_E(d) is `logZ_inf_full / P`.
- The first derivative uses the stored radial-score estimator `dlogZ_inf_full_dr / P`.
- The second derivative is a finite difference of the stored first derivative along the radius grid.
- Rules and eta flips use the same shell SMC/resampling estimator after their own dataset/reference rows are loaded.
- The NMSTV color/axis metric is recomputed from the actual direct-run `dataset_path` labels on kNN graphs with k=8,16,32.

Run status:
- `rule_units`: `12000`
- `eta_units`: `9000`
- `total_units`: `21000`
- `groups`: `7`
- `radius_min`: `0.01`
- `radius_max`: `1.0`
- `p_dim`: `2461.0`
- `dataset_metric`: `recomputed graph TV/NMSTV from each direct-run dataset_path, averaged over k=8,16,32`

Figures:
- `fig01_rules_eta_phi_energy_spaghetti.png`
- `fig02_rules_eta_direct_dphi_dd_spaghetti.png`
- `fig03_rules_eta_direct_curvature_phase_like.png`
