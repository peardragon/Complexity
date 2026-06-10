# DNN Complexity Measure

This stage computes the dataset-complexity observable used in the DNN analysis.

## Method

- complexity: normalized multiscale total variation (NMSTV)
- interpretation: graph total variation / decision-boundary stiffness with
  respect to the ground-truth labels
- graph input: cached kNN graph distances from dataset generation
- main code: `src/nmstv.py`, `src/pipeline.py`, `src/stats.py`,
  `src/visuals.py`

## Active Outputs

- `raw_outputs/18_beta_cell_30_dataset_nmstv/`: public 18-beta / 30-dataset
  complexity subset tables.
- `raw_outputs/18_beta_cell_60_dataset_30_reference/`: retained complexity
  report for the completed extension.

## Claim Boundary

NMSTV is the dataset-side complexity coordinate. It is not a landscape
observable by itself; it is compared downstream against shell entropy and proxy
local-entropy summaries.
