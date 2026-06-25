# Theory Pipeline

The theory side is the controlled validation arm of the project. It keeps the
perceptron local-entropy calculation where analytic Franz-Parisi style theory
and Monte Carlo shell estimates can be compared directly.

## Stages

- `01_theory_analytic/`: analytic replica-symmetric perceptron calculation for
  alpha = 0.1.
- `02_theory_sampling/`: two-pool perceptron shell sampling around reference
  solutions.
- `03_theory_comparison/`: final theory-vs-sampler comparison plots.

## Active Claim

The active theory claim is sampler validation, not a DNN claim. The retained
comparison figure is
`03_theory_comparison/figures/fig00_dense_qc_N_convergence_alpha0p1.png`.

The DNN stage reuses the same shell-estimation logic at the methodological
level but uses a different system and an all-SMC runtime policy. The boundary is
recorded in `../SHELL_POOL_PARALLEL_CONTRACT.md`.
