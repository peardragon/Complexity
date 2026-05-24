# Dense QC theory comparison report

이 폴더는 analytic curve와 sampling-only summary를 결합해 comparison figure를 재현하는 데 필요한 비교 테이블을 보관한다.

## Inputs

- Analytic source: `01_theory/01_theory_analytic/raw_outputs/theory_full_rs_alpha0p1.csv`
- Sampling source: `01_theory/02_theory_sampling/raw_outputs/shell_pool/sampling_phi_by_N_alpha0p1.csv`
- Sampling QC source: `01_theory/02_theory_sampling/raw_outputs/shell_pool/sampling_qc_by_N_radius.csv`

## Output files

- `comparison_phi_by_N_alpha0p1.csv`: legacy overlay table이며 analytic phi와 empirical sampling phi를 함께 담는다.
- `finiteN_error_summary.csv`: N별 sampling curve와 analytic curve의 finite-N error summary이다.
- `comparison_qc_summary.csv`: comparison figure의 QC panel을 그리기 위한 sampling QC table copy이다.
