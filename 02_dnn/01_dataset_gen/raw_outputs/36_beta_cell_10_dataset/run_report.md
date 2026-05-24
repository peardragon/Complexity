# Dataset generation run report: 36_beta_cell_10_dataset

이 폴더는 synthetic beta-cell dataset을 재현하기 위한 raw dataset payload와 dataset index를 보관한다.

## Config

```json
{
  "beta_series": [
    0.05,
    0.06,
    0.07,
    0.08,
    0.09,
    0.1,
    0.11,
    0.12,
    0.13,
    0.14,
    0.15,
    0.16,
    0.17,
    0.18,
    0.19,
    0.2,
    0.21,
    0.22,
    0.23,
    0.24,
    0.25,
    0.26,
    0.27,
    0.28,
    0.29,
    0.3,
    0.31,
    0.32,
    0.33,
    0.34,
    0.35,
    0.36,
    0.37,
    0.38,
    0.39,
    0.4
  ],
  "datasets_per_cell": 10,
  "force": true,
  "input_dim": 2,
  "ising_sweeps": 2000,
  "k_graph": 10,
  "methodology_id": "ws_ising_dataset_v1",
  "n_points": 512,
  "nmstv_scales": [
    0.5,
    1.0,
    2.0,
    4.0
  ],
  "p_series": [],
  "pipeline_id": "synthetic_dataset",
  "reuse_duplicate_cell_datasets": true,
  "rewire_mode": "degree_preserve",
  "seed": 0
}
```

## Output files

- `raw_datasets/`: cell/dataset별 `dataset.npz`와 `dataset_meta.json`을 담는 원자료 폴더이다.
- `dataset_index.csv`: raw dataset 목록, beta cell, dataset id, seed, raw/meta 경로를 담는 summary CSV이다. downstream complexity/reference stages가 이 index를 사용한다.
