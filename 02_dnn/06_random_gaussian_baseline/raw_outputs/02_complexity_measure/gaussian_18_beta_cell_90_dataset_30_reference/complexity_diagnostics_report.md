# Gaussian Random Baseline Complexity Diagnostics

Complexity proxy used here: kNN graph label roughness on normalized input features. For PM1 labels, random independent labels should have edge disagreement near 0.5 and label autocorrelation near 0.

- per-dataset table: `02_dnn/06_random_gaussian_baseline/raw_outputs/02_complexity_measure/gaussian_18_beta_cell_90_dataset_30_reference/summary_tables/dataset_complexity_per_dataset.csv`
- beta summary table: `02_dnn/06_random_gaussian_baseline/raw_outputs/02_complexity_measure/gaussian_18_beta_cell_90_dataset_30_reference/summary_tables/dataset_complexity_by_run_beta.csv`
- nearest spin beta table: `02_dnn/06_random_gaussian_baseline/raw_outputs/02_complexity_measure/gaussian_18_beta_cell_90_dataset_30_reference/summary_tables/nearest_spin_beta_to_gaussian_complexity.csv`
- Gaussian mean kNN edge disagreement: `0.500705`

Nearest spin beta tags by this proxy:

| beta | knn_edge_disagreement_mean | knn_label_autocorrelation_mean | abs_gap_to_gaussian_knn_disagreement |
|---:|---:|---:|---:|
| 0.05 | 0.478021 | 0.043959 | 0.022684 |
| 0.07 | 0.461848 | 0.076304 | 0.038857 |
| 0.09 | 0.436997 | 0.126006 | 0.063708 |
| 0.11 | 0.411254 | 0.177492 | 0.089451 |
| 0.13 | 0.378345 | 0.243309 | 0.122360 |
