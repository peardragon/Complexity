# DNN Reference Search

This stage trains the fixed 3NN architecture and selects reference solutions
around which shell-local landscape statistics are measured.

## Method

- model: `2-48-48-1-tanh`, `P=2545`
- objective: cross entropy with L2 reference selection downstream
- selected references: top exact/sampling-eligible references per dataset
- main code: `src/pipeline.py`, `src/training.py`, `src/network.py`,
  `src/rescue.py`

## Active Outputs

- `raw_outputs/18_beta_cell_30_dataset_30_reference/`: public 18-beta /
  30-dataset reference-search summaries and selected-reference payloads.
- `raw_outputs/18_beta_cell_60_dataset_30_reference/`: completed 18-beta /
  60-dataset extension summary.

## Downstream Use

Stage `04_sampling` consumes selected references through
`04_sampling/raw_outputs/reference_pool/...`, not directly from optimizer
attempt folders. The 60-dataset extension is retained as summary-only for this
stage.
