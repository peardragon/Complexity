# Reference search run report: 36_beta_cell_10_dataset_10_reference

## Config

- input: corresponding dataset generation `raw_datasets/` and `dataset_index.csv`
- reference search: per dataset/width optimizer attempts, then selected reference extraction

## Output files

- `raw_attempts/`: optimizer attempts and `attempt_results.csv`; selected references를 다시 산출하기 위한 원자료이다.
- `selected_references/`: downstream sampling이 실제로 쓰는 `selected_refs.json`, `selected_ref_payloads/`, best theta/result payload를 담는다.
- `summary_tables/`: selected-reference coverage, beta sweep status, best-per-dataset/cell summary를 담는다.
