# Advanced Refpool Visualization Report

Generated at: 2026-06-20T16:30:42

## Scope

- Source run: `refpool1024_advanced_90ref`.
- Radius grid: `0.10, 0.15, ..., 2.50` (49 radii).
- Rules: `low_tv_spectral_teacher`, `real_even_odd`, `teacher_nn`, `random_label`.
- Sampling status: complete, 90 references per rule/radius.

## Checks

- Rule/radius rows: `196`.
- Dataset embedding rows: `512` train samples.
- Reference embedding rows: `360` references.

## Tables

- `tables/dataset_train_samples_with_rule_labels.csv`
- `tables/dataset_label_summary.csv`
- `tables/dataset_tsne_umap_embedding.csv`
- `tables/advanced_phi_by_rule_radius.csv`
- `tables/advanced_dphi_dd_by_rule_radius.csv`
- `tables/reference_phi_curve_tsne_umap_embedding.csv`
- `tables/reference_phi_curve_nearest_neighbors.csv`
- `tables/reference_embedding_features.csv`
- `tables/proximal_landscape_example_refs.csv`

## Figures

- `figures/fig01_dataset_label_representatives.png`
- `figures/fig02_dataset_tsne_umap_embeddings.png`
- `figures/fig03_phi_energy_by_rule.png`
- `figures/fig04_dphi_energy_by_rule.png`
- `figures/fig05_reference_phi_curve_tsne_umap.png`
- `figures/fig06_proximal_phi_landscape_heatmap.png`
- `figures/fig07_proximal_reference_examples.png`

## Notes

- Dataset t-SNE/UMAP uses the shared 512 MNIST train images and overlays digit plus each binary rule label.
- Reference t-SNE/UMAP uses per-reference curves over raw phi, dphi/dd, and d2phi/dd2 across all 49 radii.
- The proximal landscape heatmap orders references by curve-UMAP coordinates within each rule.
