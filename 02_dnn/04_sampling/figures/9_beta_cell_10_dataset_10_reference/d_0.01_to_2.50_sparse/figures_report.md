# 04 sampling figures report: d_0.01_to_2.50_sparse

이 폴더는 DNN shell sampling stage의 active figure만 보관한다. figure는 같은 range의 summary tables를 입력으로 사용하고, raw sample payload는 summary CSV 재현용으로 유지한다.

## 입력 데이터

입력: `02_dnn/04_sampling/raw_outputs/shell_pool/9_beta_cell_10_dataset_10_reference/d_0.01_to_2.50_sparse/summary_tables/`와 필요 시 `02_dnn/04_sampling/raw_outputs/shell_pool/9_beta_cell_10_dataset_10_reference/d_0.01_to_2.50_sparse/sample_payloads/`.

## 그림 설명

- `fig_sampling_q05_ess_fraction_by_radius.png`: radius별 q=0.5 기준 effective sample size fraction을 보여주는 sampling quality figure이다.
- `fig_sampling_split_logZ_per_P_by_radius.png`: radius별 split logZ/P 추정치를 비교해 shell sampling normalizer 변화를 확인하는 그림이다.
- `fig_smc_mh_acceptance_by_radius.png`: radius별 SMC 내부 MCMC move acceptance rate를 보여주는 sampler mixing 점검 그림이다.
- `fig_smc_min_cess_by_radius.png`: radius별 SMC 최소 CESS fraction을 보여줘 tempering 안정성을 점검한다.
- `fig_smc_step_count_by_radius.png`: radius별 SMC step count를 보여줘 난이도와 compute cost 변화를 확인한다.
- `fig_weighted_accuracy_by_radius.png`: reference-weighted accuracy가 radius에 따라 어떻게 변하는지 보여준다.
- `fig_weighted_ce_l2_ratio_by_radius.png`: weighted cross-entropy term과 L2 penalty term의 비율을 radius별로 비교한다.
