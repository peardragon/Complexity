# Theory Comparison Figures Report

This folder retains active comparison figures generated from corrected
Eq. (50) theory outputs and retained sampling summaries.

## Input Data

- `01_theory/01_theory_analytic/raw_outputs/theory_full_rs_alpha0p1.csv`
- `01_theory/01_theory_analytic/raw_outputs/theory_full_feasible_rs_alpha0p1.csv`
- `01_theory/01_theory_analytic/raw_outputs/theory_full_feasible_rs_alpha0p1_fine.csv`
- `01_theory/02_theory_sampling/raw_outputs/shell_pool/sampling_phi_by_N_alpha0p1.csv`
- `01_theory/02_theory_sampling/raw_outputs/shell_pool/sampling_qc_by_N_radius.csv`
- `01_theory/03_theory_comparison/raw_outputs/dense_qc_alpha0p1/finiteN_error_summary.csv`

## Figures

- `fig00_dense_qc_N_convergence_alpha0p1.png`: promoted dense QC comparison
  figure. It overlays the corrected analytic full-RS baseline with finite-N
  PM-SAIS sampling curves and includes finite-N error and QC panels.
- `full_feasible_rs_alpha0p1/fig01_full_feasible_branch_comparison.png`:
  branch comparison for coarse and fine full-feasible grids. It shows that the
  mixed full-feasible branch coincides with the `A=0` boundary branch under the
  retained saddle-selection rule, while the max-envelope branch is diagnostic.
