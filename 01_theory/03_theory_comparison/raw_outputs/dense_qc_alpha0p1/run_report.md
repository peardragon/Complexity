# Dense QC Theory Comparison Report

This folder stores the comparison tables needed to reproduce the promoted
dense QC theory/sampling figure after the Eq. (50) quadrature correction.

## Inputs

- Analytic source:
  `01_theory/01_theory_analytic/raw_outputs/theory_full_rs_alpha0p1.csv`
- Sampling source:
  `01_theory/02_theory_sampling/raw_outputs/shell_pool/sampling_phi_by_N_alpha0p1.csv`
- Sampling QC source:
  `01_theory/02_theory_sampling/raw_outputs/shell_pool/sampling_qc_by_N_radius.csv`

## Output Files

- `comparison_phi_by_N_alpha0p1.csv`: overlay table containing corrected
  analytic `phi_theory`, empirical sampling `phi_emp`, and `finiteN_error`.
- `finiteN_error_summary.csv`: finite-N error summary against the corrected
  analytic curve.
- `comparison_qc_summary.csv`: copy of the sampling QC table used by the QC
  panel.
- `goal_status.json` and `goal_status_report.md`: regenerated comparison
  status.

## Acceptance Summary

- Goal supported: `True`
- Full QC pass: `True`
- RMSE improves from `N=40` to `N=320`: `True`
- RMSE monotone over retained N: `True`
- Smallest-N RMSE: `0.06403167224270002`
- Largest-N RMSE: `0.0018436169443736819`
- Largest-N peak radius difference: `0.0`
- QC cells: `168`
- Split-fail QC cells: `0`
- SMC-fail QC cells: `0`
