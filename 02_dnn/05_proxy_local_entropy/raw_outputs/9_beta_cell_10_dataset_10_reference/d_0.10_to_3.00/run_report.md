# Proxy local entropy raw_outputs report: d_0.10_to_3.00

이 폴더는 05_proxy_local_entropy figure를 재현하기 위한 proxy summary table을 보관한다.

## Config

- input unit summary: `02_dnn/04_sampling/raw_outputs/shell_pool/9_beta_cell_10_dataset_10_reference/d_0.10_to_3.00/summary_tables/sample_unit_summary.csv`
- input R-H summary: `02_dnn/04_sampling/raw_outputs/shell_pool/9_beta_cell_10_dataset_10_reference/d_0.10_to_3.00/summary_tables/rh_by_ref_radius_h.csv`
- proxy method: full regularized local entropy view using `logZ_inf_full`, `logZ_inf_stripped`, and `reference_prior_log_weight` from the 04 sampling unit summary
- regularization fallback: `compute_phi.DEFAULT_LAMBDA_REG=220` only if the full/correction fields are absent
- q values: 0.5, 0.9, 0.99
- accuracy quantile table regenerated: no

## Output files

- `summary_tables/absolute_phi_by_beta_radius.csv`: 04 sampling unit summary에서 beta/radius별 absolute full phi, energy term, area term을 집계한 table이다.
- `summary_tables/delta_phi_by_beta_radius.csv`: 기준 radius 대비 delta phi와 energy/area split을 beta/radius별로 집계한 table이다.
- `summary_tables/dphi_dr_by_beta_radius.csv`: radial derivative 관련 proxy quantity를 beta/radius별로 정리한 table이다.
- `summary_tables/hq_by_beta_radius.csv`: `rh_by_ref_radius_h.csv`에서 q별 H threshold phase map 입력을 만든 table이다.
- `summary_tables/accuracy_q_by_beta_radius.csv`: sample payload NPZ의 per-sample error/log weight를 다시 읽어 q별 weighted accuracy cutoff를 만든 table이다. 이 table은 `--include-accuracy` 실행 때만 생성한다.

## Reproduction chain

`04_sampling/.../summary_tables/`와 필요 시 `04_sampling/.../sample_payloads/`를 입력으로 이 폴더의 proxy summary tables를 만들고, `05_proxy_local_entropy/figures/`의 local entropy, derivative, phase-map figure가 이 table들을 사용한다.
