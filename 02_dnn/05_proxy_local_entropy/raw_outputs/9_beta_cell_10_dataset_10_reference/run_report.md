# Proxy local entropy raw_outputs report

이 폴더는 05_proxy_local_entropy stage의 active proxy summary table을 range별로 보관한다.

## Config

- source sampling root: `02_dnn/04_sampling/raw_outputs/shell_pool/9_beta_cell_10_dataset_10_reference/`
- output root: `02_dnn/05_proxy_local_entropy/raw_outputs/9_beta_cell_10_dataset_10_reference/`
- retained ranges: `d_0.01_to_0.10`, `d_0.10_to_3.00`, `d_0.01_to_2.50_dense`, `d_0.01_to_2.50_sparse`
- table builder: `02_dnn/05_proxy_local_entropy/src/make_proxy_tables.py`

## Output files

- `{range}/summary_tables/absolute_phi_by_beta_radius.csv`: beta/radius별 absolute full phi와 energy/area term summary이다.
- `{range}/summary_tables/delta_phi_by_beta_radius.csv`: 기준 radius 대비 delta phi summary이다.
- `{range}/summary_tables/dphi_dr_by_beta_radius.csv`: radial derivative proxy summary이다.
- `{range}/summary_tables/hq_by_beta_radius.csv`: q별 H threshold phase-map 입력 table이다.
- `{range}/run_report.md`: 해당 range의 config, 입력, output 설명이다.

## Accuracy quantile table

`accuracy_q_by_beta_radius.csv`는 04 sampling의 `sample_payloads/*.npz`를 다시 읽어야 하는 full recovery table이다. 원자료는 보존되어 있으나, 이번 compact 정리에서는 긴 재실행을 피하기 위해 기본 생성에서 제외했다. 필요하면 `make_proxy_tables.py --include-accuracy`로 재생성한다.
