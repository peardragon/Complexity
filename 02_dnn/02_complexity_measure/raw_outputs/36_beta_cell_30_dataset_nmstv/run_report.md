# Complexity measure run report: 36_beta_cell_30_dataset_nmstv

## Config

- input: matching `02_dnn/01_dataset_gen/raw_outputs/*/raw_datasets/` and `dataset_index.csv`
- metric: nMSTV and multiscale complexity summaries

## Output files

- `summary_tables/complexity_summary_by_dataset.csv`: dataset별 nMSTV summary이다.
- `summary_tables/complexity_summary_by_cell.csv`: beta cell별 nMSTV 평균/표준편차 summary이다.
- `summary_tables/complexity_multiscale_by_dataset.csv`: dataset별 scale-dependent complexity raw summary이다.
- `summary_tables/complexity_multiscale_by_cell.csv`: cell별 scale-dependent complexity summary이다.
- `summary_tables/complexity_multiscale_ordering_pairs.csv`: beta/scale ordering pair table이다.
- `summary_tables/complexity_multiscale_ordering_summary.json`: ordering/crossing summary이다.
- `summary_tables/complexity_series_correlations.json`: beta series correlation summary이다.
