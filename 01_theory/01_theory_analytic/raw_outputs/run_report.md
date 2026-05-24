# Analytic theory raw output report

## Config

- 설정 파일: `01_theory/01_theory_analytic/config/default.json`
- alpha: `0.1`
- lambda_ref/lambda_shell: `1` / `1`

## Output files

- `theory_full_rs_alpha0p1.csv`: analytic full-RS 해로 계산한 d별 phi 원천 표이다. `phi_rel` 열은 첫 radius 기준 상대 phi이며 analytic figure와 comparison stage가 사용한다.

## Reproduction chain

`src/theory_full_rs.py`가 config의 alpha/lambda와 radius grid를 사용해 CSV를 만들고, `figures/fig01_phi_by_analytic_solution_alpha0p1.png`가 이 CSV를 직접 읽어 생성된다.
