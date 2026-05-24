# Theory sampling figures report

이 폴더는 sampling-only active figure 산출물을 보관한다.

## 입력 데이터

입력: `01_theory/02_theory_sampling/raw_outputs/shell_pool/sampling_phi_by_N_alpha0p1.csv`.

이 CSV는 shell sampling payload에서 집계한 finite-N empirical phi(d) 결과만 담는다. analytic solution curve는 이 stage의 figure 입력으로 복제하지 않으며, analytic comparison이 필요할 때는 `01_theory/03_theory_comparison`이 analytic CSV를 별도로 읽는다.

## 그림 설명

- `fig01_sampling_phi_by_distance.png`: N별 shell sampling empirical phi(d)-phi(d0) 곡선이다. analytic curve는 포함하지 않고, sampling 결과의 거리별 변화와 N에 따른 수렴 경향만 보여준다.
