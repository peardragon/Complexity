# DNN Dataset Generation

This stage builds the synthetic 2D datasets used by the 3NN landscape
experiments.

## Method

- data source: 2D Ising spin configurations over beta cells
- labels: spin-derived ground truth used to define the decision boundary
- graph support: kNN graph metadata used downstream by NMSTV complexity
- main code: `src/pipeline.py`, `src/ising.py`, `src/dataset_builder.py`,
  `src/graphs.py`

## Active Outputs

- `raw_outputs/18_beta_cell_30_dataset/`: public 18-beta / 30-dataset index
  for the production run. Raw dataset payloads are materialized under this
  active run root.
- `raw_outputs/18_beta_cell_60_dataset/`: completed 18-beta / 60-dataset
  extension with 30 reused and 30 newly generated datasets per beta.

## Notes

The public 18-beta dataset roots are the active dataset payload surface for
downstream stages. Superseded source sweeps are archived under `99_backup/`.
