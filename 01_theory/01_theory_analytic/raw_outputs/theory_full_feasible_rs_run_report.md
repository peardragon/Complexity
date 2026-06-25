# Full Feasible RS Run Report

## Goal

Regenerate the retained theory outputs after correcting the Eq. (50)
`z0` quadrature normalization, then verify consistency between the baseline
`A=0` solver, the full-feasible solver, and the retained PM-SAIS sampling
comparison.

## Implementation

- Baseline solver: `01_theory/01_theory_analytic/src/theory_full_rs.py`
- Full-feasible solver: `01_theory/01_theory_analytic/src/theory_full_rs_feasible.py`
- Full-feasible comparison: `01_theory/03_theory_comparison/src/compare_full_feasible_rs.py`
- Full-feasible visualization:
  `01_theory/03_theory_comparison/src/plot_full_feasible_rs_comparison.py`

The feasible parametrization uses

- `t = q cd + s sqrt(q(1-q)(1-cd^2))`
- `p_min = cd^2 + s^2(1-cd^2)`
- `p = p_min + eta(1-p_min)`
- `A = eta(1-p_min)`

The energetic quadrature now follows Eq. (50): the selected-reference average
is conditional at fixed `z0`, and the Gaussian weight `Dz0` is applied exactly
once outside that conditional average.

## Runtime Checks

Python used for runs:

```text
D:\Complexity\.venv\Scripts\python.exe
```

Numba was available in the project virtual environment.

Smoke and timing checks:

| run | grid | radii | wall/solver time |
| --- | --- | ---: | ---: |
| corrected smoke | q=15, s=11, eta=5 | 3 | 2.951 s wall |
| one-radius coarse timing | q=45, s=41, eta=21 | 1 | 8.887 s wall |
| one-radius fine timing | q=75, s=61, eta=31 | 1 | 14.888 s wall |

Estimated runtime from one-radius timing was sub-day, so the full coarse and
fine runs were launched as background processes and monitored through logs.

Actual solver log elapsed:

| run | grid | radii | elapsed |
| --- | --- | ---: | ---: |
| baseline | q=45, s=31 | 42 | 6.010 s wall |
| coarse | q=45, s=41, eta=21, eta_max=0.98 | 42 | 201.8 s |
| fine | q=75, s=61, eta=31, eta_max=0.995 | 42 | 469.8 s |

The coarse and fine full-feasible runs were executed concurrently; the fine
elapsed time includes CPU contention before the coarse run finished.

## Outputs

Baseline analytic:

- `01_theory/01_theory_analytic/raw_outputs/theory_full_rs_alpha0p1.csv`

Coarse analytic:

- `01_theory/01_theory_analytic/raw_outputs/theory_full_feasible_rs_alpha0p1.csv`

Coarse comparison:

- `01_theory/03_theory_comparison/raw_outputs/full_feasible_rs_alpha0p1/comparison_phi_full_feasible_by_N_alpha0p1.csv`
- `01_theory/03_theory_comparison/raw_outputs/full_feasible_rs_alpha0p1/finiteN_error_full_feasible_summary.csv`
- `01_theory/03_theory_comparison/raw_outputs/full_feasible_rs_alpha0p1/branch_A_eta_diagnostics.csv`
- `01_theory/03_theory_comparison/raw_outputs/full_feasible_rs_alpha0p1/full_feasible_goal_status.json`

Fine analytic:

- `01_theory/01_theory_analytic/raw_outputs/theory_full_feasible_rs_alpha0p1_fine.csv`

Fine comparison:

- `01_theory/03_theory_comparison/raw_outputs/full_feasible_rs_alpha0p1_fine/comparison_phi_full_feasible_by_N_alpha0p1.csv`
- `01_theory/03_theory_comparison/raw_outputs/full_feasible_rs_alpha0p1_fine/finiteN_error_full_feasible_summary.csv`
- `01_theory/03_theory_comparison/raw_outputs/full_feasible_rs_alpha0p1_fine/branch_A_eta_diagnostics.csv`
- `01_theory/03_theory_comparison/raw_outputs/full_feasible_rs_alpha0p1_fine/full_feasible_goal_status.json`

Visualization:

- `01_theory/03_theory_comparison/figures/full_feasible_rs_alpha0p1/fig01_full_feasible_branch_comparison.png`

## Acceptance Checks

Eq. (50) quadrature checks:

- Source search found no duplicated `z0` weight pattern.
- Constant-integrand Eq. (50) normalization: `1.0016879917552934`
- Legacy duplicated-`z0` normalization on the same check: `0.09896283472721967`
- Baseline/full-feasible `A=0` energetic-term absolute difference:
  `5.551115123125783e-17`

Artifact integrity:

- Baseline rows: `42`
- Coarse full-feasible rows: `126`
- Fine full-feasible rows: `126`
- All three theory stderr logs have size `0`.

Coarse baseline reproduction:

- Shared radii: `42`
- Max absolute `phi_rel` difference to corrected baseline:
  `2.4559543770008574e-05`
- RMSE difference to corrected baseline: `9.895039011598244e-06`
- Passes `1e-3`, `2e-3`, and `3e-3`

Fine baseline reproduction against corrected coarse baseline:

- Shared radii: `42`
- Max absolute `phi_rel` difference: `0.0026816052756530873`
- RMSE difference: `0.0006369226360866918`
- Passes `3e-3`; the difference is the expected grid-refinement effect.

## Main Result

Both coarse and fine full mixed saddles collapse to the boundary:

| run | branch | max_A | max_eta | interior A>1e-4 radii |
| --- | --- | ---: | ---: | ---: |
| coarse | full_mixed_maxQ_min_s_eta | 0.0 | 0.0 | 0 |
| fine | full_mixed_maxQ_min_s_eta | 0.0 | 0.0 | 0 |

The diagnostic full max envelope selects interior eta at every radius:

| run | branch | max_A | max_eta | mean_A |
| --- | --- | ---: | ---: | ---: |
| coarse | full_max_envelope | 0.0955473166019208 | 0.98 | 0.06453563722699313 |
| fine | full_max_envelope | 0.0651922365511097 | 0.995 | 0.044166957011978036 |

Largest-N comparison against PM-SAIS:

| run | N | branch | RMSE | peak radius abs diff |
| --- | ---: | --- | ---: | ---: |
| coarse | 320 | boundary_mixed_eta0 | 0.001846035835878348 | 0.0 |
| coarse | 320 | full_mixed_maxQ_min_s_eta | 0.001846035835878348 | 0.0 |
| coarse | 320 | full_max_envelope | 0.008008664594284107 | 0.0 |
| fine | 320 | boundary_mixed_eta0 | 0.0017405702323724564 | 0.0 |
| fine | 320 | full_mixed_maxQ_min_s_eta | 0.0017405702323724564 | 0.0 |
| fine | 320 | full_max_envelope | 0.007631122392776281 | 0.0 |

Interpretation:

The corrected Eq. (50) normalization substantially improves agreement with the
retained PM-SAIS sampling comparison. Under the retained mixed saddle selection
the full-feasible branch selects `eta_star = 0` and `A_star = 0` across the
tested radii, so the physical mixed branch coincides with the `A=0` boundary in
this regime. The pure max envelope remains a diagnostic upper envelope and
should not be treated as the physical branch without additional saddle-selection
analysis.
