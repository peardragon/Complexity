# Advanced Refpool Visualization Report

Generated at: 2026-06-26T00:24:49

## Scope

- Source run: `very_low_tv_spectral_teacher_refpool1024_advanced_90ref`.
- Radius grid: `0.10, 0.15, ..., 2.50` (49 radii).
- Rules: `very_low_tv_spectral_teacher`, `real_even_odd`, `teacher_nn`, `random_label`.
- Sampling status: complete, 90 references per rule/radius.
- Dataset representations are stored under `01_dataset_gen`.
- Phi/dphi and reference-curve representations are stored under `05_proxy_local_entropy`.

## Checks

- Rule/radius rows: `196`.
- Dataset embedding rows: `512` train samples.
- Reference embedding rows: `360` references.

## Tables

- `01_dataset_gen/raw_outputs/active_rule_dataset_representations_very_low_refpool1024_advanced_90ref/tables/dataset_train_samples_with_rule_labels.csv`
- `01_dataset_gen/raw_outputs/active_rule_dataset_representations_very_low_refpool1024_advanced_90ref/tables/dataset_label_summary.csv`
- `01_dataset_gen/raw_outputs/active_rule_dataset_representations_very_low_refpool1024_advanced_90ref/tables/dataset_tsne_umap_embedding.csv`

## Figures

- `01_dataset_gen/figures/active_rule_dataset_representations_very_low_refpool1024_advanced_90ref/fig01_dataset_label_representatives.png`
- `01_dataset_gen/figures/active_rule_dataset_representations_very_low_refpool1024_advanced_90ref/fig02_dataset_tsne_umap_embeddings.png`

## Notes

- Dataset t-SNE/UMAP uses the shared 512 MNIST train images and overlays digit plus each binary rule label.
- Reference t-SNE/UMAP uses per-reference curves over raw phi, dphi/dd, and d2phi/dd2 across all 49 radii.
- The proximal landscape heatmap orders references by curve-UMAP coordinates within each rule.

