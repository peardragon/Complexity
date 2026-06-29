# Derivative Stability And Spin Comparison

This analysis reads existing completed CSVs only. It does not rerun reference search or sampling.

## Main Findings

1. `phi_E(d)` is the primary MNIST observable and is stable at the mean-curve level.
2. Current MNIST `dphi_E/dd` is not a direct sampled derivative. It is reconstructed from `phi_E(d)` by finite differences, so it is sensitive to radius step, smoothing, and edge handling.
3. The positive-curvature mass `A_kappa` is useful as a descriptive diagnostic, but its absolute value is method dependent in MNIST.
4. The 3NN spin result is methodologically stronger for phase-transition claims because the first derivative was stored directly by the sampler and aggregated over a much larger beta/radius/reference table.
5. The MNIST eta/rule curves support a smooth crossover interpretation rather than a sharp order-parameter collapse.

## Estimator Sensitivity Snapshot

| case | raw A_kappa | SG21 A_kappa | raw sign changes | SG21 sign changes |
| ---- | ----------- | ------------ | ---------------- | ----------------- |
| even odd | 4.207 | 0.1682 | 43.53 | 14.23 |
| eta=0.02 | 4.592 | 0.1992 | 43.83 | 13.63 |
| eta=0.05 | 4.784 | 0.2641 | 44.10 | 11.23 |
| eta=0.15 | 5.903 | 0.4605 | 44.70 | 10.53 |
| eta=0.25 | 6.768 | 0.6073 | 45.83 | 11.07 |

The raw-gradient curvature contains many sign changes because a 0.01-spaced finite difference amplifies small SMC/logZ fluctuations. Smoothing or coarsening preserves the broad ordering by eta but changes `A_kappa` scale and peak locations.

## Spin Versus MNIST

- Spin `A_kappa` ranges from 0.00638834 to 0 and collapses to zero across the beta sweep.
- MNIST `A_kappa` in the SG21 diagnostic ranges from 0.1682 to 0.6073, but it is already positive at the lowest eta/even-odd cases and changes gradually.
- Therefore the MNIST evidence is better described as a smooth family-dependent crossover in local free energy, not a sharp phase transition.

## Noise/QC Correlations

- eta_0.02: corr(split_mean, curvature_abs_mean) = 0.349
- eta_0.02: corr(split_fail_frac, curvature_abs_mean) = 0.330
- eta_0.02: corr(phi_sem, curvature_abs_mean) = 0.318
- eta_0.02: corr(ess_fraction_mean, curvature_abs_mean) = -0.196
- eta_0.05: corr(split_mean, curvature_abs_mean) = 0.209
- eta_0.05: corr(split_fail_frac, curvature_abs_mean) = 0.234
- eta_0.05: corr(phi_sem, curvature_abs_mean) = 0.132
- eta_0.05: corr(ess_fraction_mean, curvature_abs_mean) = -0.157
- eta_0.15: corr(split_mean, curvature_abs_mean) = -0.065
- eta_0.15: corr(split_fail_frac, curvature_abs_mean) = -0.033
- eta_0.15: corr(phi_sem, curvature_abs_mean) = -0.086
- eta_0.15: corr(ess_fraction_mean, curvature_abs_mean) = -0.054
- eta_0.25: corr(split_mean, curvature_abs_mean) = 0.036
- eta_0.25: corr(split_fail_frac, curvature_abs_mean) = 0.093
- eta_0.25: corr(phi_sem, curvature_abs_mean) = 0.062
- eta_0.25: corr(ess_fraction_mean, curvature_abs_mean) = -0.107
- even_odd: corr(split_mean, curvature_abs_mean) = 0.563
- even_odd: corr(split_fail_frac, curvature_abs_mean) = 0.553
- even_odd: corr(phi_sem, curvature_abs_mean) = 0.448
- even_odd: corr(ess_fraction_mean, curvature_abs_mean) = -0.253

## Outputs

- `fig05_derivative_estimator_stability.png`
- `fig06_qc_noise_derivative_diagnostics.png`
- `fig07_spin_synthetic_vs_mnist_real_phase_comparison.png`
- `derivative_stability_summary_by_case_method.csv`
- `qc_noise_derivative_spike_by_case_radius.csv`
- `qc_noise_derivative_spike_correlations.csv`
- `phase_metric_comparison_spin_vs_mnist.csv`

## Interpretation For Paper Discussion

For MNIST, we should claim robust ordering and smooth crossover in `phi_E(d)` as eta/rule complexity increases. We should not claim a spin-like phase transition from the current derivative/curvature diagnostics alone. To make a stronger phase-transition claim on MNIST, the next necessary experiment is either a direct derivative estimator analogous to the 3NN spin stack, or independent replicate sampling at selected radii combined with a pre-registered smoothing/coarsening rule.
