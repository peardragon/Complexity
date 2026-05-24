# 04 sampling shell_pool run report: d_0.01_to_0.10

## Config

- dataset input: `02_dnn/01_dataset_gen/raw_outputs/36_beta_cell_10_dataset/raw_datasets/`
- reference input: `02_dnn/04_sampling/raw_outputs/reference_pool/9_beta_cell_10_dataset_10_reference/selected_reference_pool/`
- sampler: exact shell L2 VMF adaptive CE-tempered SMC
- samples per reference/radius: 1024 for the retained final ranges

## Output files

- `sample_payloads/`: cell/dataset/reference/radius별 raw sampling payload이다. summary CSV와 proxy local entropy figures를 재현하는 원자료이다.
- `summary_tables/sample_unit_summary.csv`: raw sampling unit-level summary이다.
- `summary_tables/beta_radius_summary.csv`: beta/radius aggregate summary이다.
- `summary_tables/rh_by_ref_radius_h.csv`: reference/radius/h별 R-h summary이다.
- `summary_tables/loss_gap_ratios.csv`: loss gap ratio summary이다.
- `summary_tables/radial_derivative_qc_by_beta_radius.csv`: radial derivative QC summary이다.
- `summary_tables/failed_units.json`: 실패 unit summary이다.

## Reproduction chain

`sample_payloads/`를 집계해 `summary_tables/`의 CSV/JSON을 만들고, `figures/` 및 `05_proxy_local_entropy/figures/`가 이 summary tables를 입력으로 사용한다.
