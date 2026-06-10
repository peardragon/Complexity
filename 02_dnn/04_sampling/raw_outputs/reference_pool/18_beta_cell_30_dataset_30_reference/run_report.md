# Sampling reference-pool report: 18_beta_cell_30_dataset_30_reference

This is the public 18-beta / 30-dataset reference pool used by the promoted
production shell run.

- active subset manifest: `selected_reference_pool/final_pool1_l2_top30_refs.json`
- selected beta cells: `0.05, 0.07, ..., 0.39`
- dataset pools: `540`
- references per dataset: `30`
- payload policy: selected reference summaries and theta payloads are
  materialized under `selected_reference_pool/`; the manifest path fields point
  to this active run root.
