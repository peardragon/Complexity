# Theory comparison figures report

이 폴더는 active figure 산출물만 보관한다. 각 그림은 아래 입력 데이터에서 재생성된다.

## 입력 데이터

입력:

- `01_theory/01_theory_analytic/raw_outputs/theory_full_rs_alpha0p1.csv`
- `01_theory/02_theory_sampling/raw_outputs/shell_pool/sampling_phi_by_N_alpha0p1.csv`
- `01_theory/02_theory_sampling/raw_outputs/shell_pool/sampling_qc_by_N_radius.csv`
- `01_theory/03_theory_comparison/raw_outputs/dense_qc_alpha0p1/finiteN_error_summary.csv`

## 그림 설명

- `fig00_dense_qc_N_convergence_alpha0p1.png`: analytic full-RS phi(d)와 sampling-only finite-N phi(d)를 같은 좌표계에서 비교하고, 아래 패널에 finite-N error와 QC 통과율을 함께 보여주는 promoted comparison figure이다. analytic curve 데이터는 sampling raw_outputs에 복제하지 않고 analytic stage CSV에서 직접 읽는다.
