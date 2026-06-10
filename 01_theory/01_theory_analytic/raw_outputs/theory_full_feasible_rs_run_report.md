# Full Feasible RS Run Report

## Goal

Verify whether the PM-SAIS shell sampling agreement supports only the retained
A=0 boundary branch or the full feasible-domain RS solver with A >= 0.

## Implementation

- Added solver: `01_theory/01_theory_analytic/src/theory_full_rs_feasible.py`
- Added comparison: `01_theory/03_theory_comparison/src/compare_full_feasible_rs.py`
- Added visualization: `01_theory/03_theory_comparison/src/plot_full_feasible_rs_comparison.py`
- Existing baseline was not overwritten:
  `01_theory/01_theory_analytic/src/theory_full_rs.py`
- `01_theory/01_theory_analytic_revised` was not created or modified.

The feasible parametrization uses

- `t = q cd + s sqrt(q(1-q)(1-cd^2))`
- `p_min = cd^2 + s^2(1-cd^2)`
- `p = p_min + eta(1-p_min)`
- `A = eta(1-p_min)`

The energetic quadrature extends the retained baseline solver's legacy
normalization convention so that eta=0 is directly comparable to the retained
`theory_full_rs_alpha0p1.csv`.

## Runtime Checks

Python used for full runs:

```text
D:\Complexity\.venv\Scripts\python.exe
```

The default Anaconda Python has a Numba/NumPy ABI mismatch, so the full runs
used the project `.venv`, where Numba imports successfully.

Smoke and timing checks:

| run | grid | radii | wall/solver time |
| --- | --- | ---: | ---: |
| smoke compile/path | q=5, s=5, eta=3 | 2 | 14.2 s wall |
| smoke timing | q=15, s=11, eta=5 | 3 | 4.6 s wall |
| one-radius coarse timing | q=45, s=41, eta=21 | 1 | 6.9 s wall |
| one-radius fine timing | q=75, s=61, eta=31 | 1 | 13.9 s wall |

Estimated runtime from one-radius timing was sub-day, so the full coarse and
fine runs were launched as background processes and monitored through logs.

Actual solver log elapsed:

| run | grid | radii | elapsed |
| --- | --- | ---: | ---: |
| coarse | q=45, s=41, eta=21, eta_max=0.98 | 42 | 109.1 s |
| fine | q=75, s=61, eta=31, eta_max=0.995 | 42 | 387.4 s |

## Outputs

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

Coarse baseline reproduction:

- Shared radii: 42
- Max absolute `phi_rel` difference to retained baseline: `1.8173920152353418e-05`
- RMSE difference to retained baseline: `9.407441005055805e-06`
- Passes `1e-3` and `2e-3`

Fine baseline reproduction against retained coarse baseline:

- Shared radii: 42
- Max absolute `phi_rel` difference: `0.0025639745020664684`
- RMSE difference: `0.0007680553410447146`
- Passes `3e-3`, not `2e-3`

The fine-grid difference is a grid refinement effect, not a feasible-solver
implementation mismatch. Recomputing the old `theory_full_rs.py` logic on the
same fine grid gives:

- Wide-s same-grid max absolute difference: `4.440892098500626e-16`
- Legacy-s same-grid max absolute difference: `8.869266898248185e-06`

## Main Result

Both coarse and fine full mixed saddles collapse to the boundary:

| run | branch | max_A | max_eta | interior A>1e-4 radii |
| --- | --- | ---: | ---: | ---: |
| coarse | full_mixed_maxQ_min_s_eta | 0.0 | 0.0 | 0 |
| fine | full_mixed_maxQ_min_s_eta | 0.0 | 0.0 | 0 |

The diagnostic full max envelope selects interior eta at every radius:

| run | branch | max_A | max_eta | mean_A |
| --- | --- | ---: | ---: | ---: |
| coarse | full_max_envelope | 0.0955473166019208 | 0.98 | 0.06450427669928242 |
| fine | full_max_envelope | 0.0651922365511097 | 0.995 | 0.0440866255935805 |

Largest-N comparison against PM-SAIS:

| run | N | branch | RMSE | peak radius abs diff |
| --- | ---: | --- | ---: | ---: |
| coarse | 320 | full_max_envelope | 0.020037196349409727 | 0.0 |
| coarse | 320 | full_mixed_maxQ_min_s_eta | 0.020684723993786854 | 0.0 |
| coarse | 320 | boundary_mixed_eta0 | 0.020684723993786854 | 0.0 |
| fine | 320 | full_max_envelope | 0.020471029506922207 | 0.0 |
| fine | 320 | full_mixed_maxQ_min_s_eta | 0.021076770741890495 | 0.0 |
| fine | 320 | boundary_mixed_eta0 | 0.021076770741890495 | 0.0 |

Interpretation:

The physical mixed saddle under the retained sign convention selects
`eta_star = 0` and `A_star = 0` across the tested radii. PM-SAIS therefore
supports the full feasible-domain RS formula in this tested regime with the
selected branch lying on the boundary of the covariance cone. The pure max
envelope is a diagnostic upper envelope and should not be treated as the
physical branch without further saddle-selection analysis.

The branch comparison figure confirms the same result visually: the
`full_mixed` curve overlays the `boundary_mixed_eta0` curve, the
`full_mixed - boundary` delta is zero, and the only visible A>0 curve is the
diagnostic `full_max_envelope`.
