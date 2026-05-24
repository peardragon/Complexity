# Analytic theory figures report

이 폴더는 analytic solution만 사용한 active figure 산출물을 보관한다.

## 입력 데이터

입력: `01_theory/01_theory_analytic/raw_outputs/theory_full_rs_alpha0p1.csv`.

이 CSV는 alpha=0.1 full-RS analytic solution에서 radius `r`, raw objective `phi`, normalized value `phi_rel`, 그리고 최적화 보조 변수 `Q`, `p`, `t`, `cd`, `s`, `qref`를 보관한다.

## 그림 설명

- `fig01_phi_by_analytic_solution_alpha0p1.png`: analytic full-RS solution으로 계산한 phi(d)-phi(d0) 곡선이다. sampling 결과나 finite-N empirical curve를 함께 그리지 않으며, analytic stage의 canonical CSV만 사용한다.
