# Theory Analytic

This stage computes the analytic perceptron local-entropy reference used to
validate the Monte Carlo shell sampler.

## Method

- system: L2-regularized perceptron
- setting: alpha = 0.1
- observable: analytic Franz-Parisi style local entropy curve
- main implementation: `src/theory_full_rs.py`
- plotting helpers: `src/make_phi_figure.py`, `src/plotting.py`

## Active Outputs

- `raw_outputs/theory_full_rs_alpha0p1.csv`: analytic curve values
- `figures/`: retained analytic figures and `figures_report.md`

This stage is upstream of `../03_theory_comparison/`.
