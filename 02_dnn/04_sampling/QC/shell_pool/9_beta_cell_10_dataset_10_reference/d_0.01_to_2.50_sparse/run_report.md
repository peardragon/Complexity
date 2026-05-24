# 04 sampling QC report: d_0.01_to_2.50_sparse

이 폴더는 DNN shell sampling range의 QC summary만 보관한다. legacy run_config와 manifest는 active 재현 입력이 아니라 provenance 성격이므로 `99_backup/cleanup_20260524_075739/` 아래로 이동했다.

## 입력 데이터

입력: `02_dnn/04_sampling/raw_outputs/shell_pool/9_beta_cell_10_dataset_10_reference/d_0.01_to_2.50_sparse/summary_tables/`와 compact config의 sampling settings.

## Output files

- `sampling_qc_by_beta_radius.csv`: beta/radius별 sampling QC pass/fail 및 주요 sampler quality metric 요약이다.
- `claim_gate_summary.csv`: retained figure 또는 proxy claim에 필요한 gate status를 요약한다.
- `radial_derivative_qc_by_beta_radius.csv`: radial derivative 관련 QC metric을 beta/radius별로 정리한다.
