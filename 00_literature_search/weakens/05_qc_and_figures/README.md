# QC And Figures

Main outputs:

- `raw_outputs/<experiment_id>/region_mass_estimates.csv`
- `raw_outputs/<experiment_id>/qc_summary.csv`
- `raw_outputs/<experiment_id>/qc_report.json`
- `figures/<experiment_id>/final_sampling_failure_vmf_recovery.png`
- `figures/<experiment_id>/final_sampling_failure_vmf_recovery_AB.png`
- `figures/<experiment_id>/schematic_problem_statement_vmf_l2.png`
  - top row: simplified problem geometry with minimal in-panel labels;
  - bottom row: indirect trajectory/effect view from retained samples, vMF
    proposal footprint, and region-coverage accumulation.
- `figures/method_sampling_results.ipynb`
  - creates five plots under `figures/<experiment_id>/method_sampling_results/`:
    Figure A for the proxy landscape plus Figure B1-B4 for method-wise
    sampling results;
  - also writes `diagnostic_computed_barrier_guides.png`, which shows the
    computed minimax paths and extracted high-energy barrier windows;
  - HMC and pL are shown as sampling trajectories, while MC and vMF+L2 are
    shown as point samples.
- `figures/method_sampling_results_l2_common.ipynb`
  - same figure protocol for `default_l2_common.json`;
  - all methods share the `E_proxy(z) + lambda ||z||^2` target;
  - the notebook displays attempted sample counts and elapsed seconds from
    `qc_summary.csv`.

QC mirrors the `../../02_dnn/04_sampling` style: region mass, hit counts, ESS
fraction, and weighted estimates are more important than visually pleasing
scatter alone.

The default generated result currently has the intended separation:

- `random_walk_mcmc`, `hmc`, and `pseudo_langevin` fail QC because they miss at
  least one important region.
- `vmf_l2_final` passes QC as the final-method proxy instantiation.
