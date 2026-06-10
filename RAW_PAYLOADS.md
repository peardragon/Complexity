# Raw Payload Policy

The active workspace keeps promoted raw payloads locally, but Git tracks only
the compact reproducibility surface:

- configs
- retained source code
- metadata tables
- manifests and provenance
- figures
- smoke recovery outputs when explicitly promoted

Large binary payload trees are intentionally ignored by Git because the active
workspace can contain millions of `.npz`/`.npy` files. Tracking those directly
would make normal Git operations and uploads impractical even with Git LFS.

## Active Local Payload Roots

- `01_theory/02_theory_sampling/raw_outputs/shell_pool/sample_payloads/`
- `02_dnn/01_dataset_gen/raw_outputs/18_beta_cell_30_dataset/raw_datasets/`
- `02_dnn/01_dataset_gen/raw_outputs/18_beta_cell_60_dataset/raw_datasets/`
- `02_dnn/03_reference_search/raw_outputs/18_beta_cell_30_dataset_30_reference/selected_references/`
- `02_dnn/04_sampling/raw_outputs/reference_pool/18_beta_cell_30_dataset_30_reference/selected_reference_pool/cell_beta_*/`
- `02_dnn/04_sampling/raw_outputs/shell_pool/18_beta_cell_30_dataset_30_reference/d_0.01_to_2.50_dense/sample_payloads/`

The public `18_beta_cell_30_dataset_30_reference` dataset/reference subset is
materialized under active 18-beta paths.

## Summary-Only Active Roots

The completed `18_beta_cell_60_dataset_30_reference` extension retains compact
reference, shell, and proxy summaries plus figures. It does not retain raw
theta payloads or row-level shell sample payloads in the active public path.

## Cleanup Note

Superseded 9-beta/10-dataset outputs, non-active 36-beta outputs, and DNN smoke
run payloads were archived under `99_backup/cleanup_20260610_141822/` and
`99_backup/cleanup_20260610_150752/`. The cleanup manifests also record the
blocked `02_dnn/transfer_archives/` move caused by a locked temporary archive
file.
