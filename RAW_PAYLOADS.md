# Raw Payload Policy

The active directory layout keeps promoted raw payloads on the workstation, but
the GitHub repository tracks the compact reproducibility surface:

- configs
- retained source code
- metadata tables
- manifests and provenance
- figures
- smoke recovery outputs

Large binary raw payload trees are intentionally ignored by Git because the
current compact workspace contains more than one million `.npz`/`.npy` payload
files. Tracking those directly would make normal Git operations and GitHub
uploads impractical even with Git LFS.

Ignored local payload roots and row-level metadata:

- `01_theory/02_theory_sampling/raw_outputs/shell_pool/sample_payloads/`
- `02_dnn/01_dataset_gen/raw_outputs/**/raw/`
- `02_dnn/02_complexity_measure/raw_outputs/**/raw/`
- `02_dnn/03_reference_search/raw_outputs/**/raw/`
- `02_dnn/04_sampling/raw_outputs/**/raw/`
- `02_dnn/04_sampling/raw_outputs/reference_pool/*/selected_reference_pool/cell_beta_*/`
- dense/sparse shell sampling row-level CSV/JSONL metadata under
  `02_dnn/04_sampling/raw_outputs/shell_pool/*/d_0.01_to_2.50_*/metadata/`

The payloads remain present locally after the cleanup. The tracked metadata and
provenance files record the promoted source identities and user-friendly active
paths.
