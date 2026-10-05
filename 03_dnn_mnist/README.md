# MNIST: Label Noise and Digit-Pair Experiments

MNIST comparisons for Figs. 5–8.

## Project Overview

- **Preprocessing**: 28×28 images reduced to 10×10 by box averaging, with training-set standardization.
- **Model**: 100–20–20–1 tanh network, P=2461.
- **Replicas**: Ten datasets and ten references per condition.
- **Sampling**: r=0.01–1.00; two independent pools of 512 particles.

## Directory Structure

```text
03_dnn_mnist/
├── label_noise_sweep/           # Eta=0,.05,.15,.25,.5
└── digit_pairwise_complexity/   # Twelve selected digit pairs
    ├── 01_dataset/
    ├── 02_complexity_measure/
    ├── 03_reference_search/
    ├── 04_sampling/
    └── 05_proxy_local_entropy/
```

Both experiments use the five-stage structure of the synthetic experiment.

## Workflow & Dependencies

```bash
python 03_dnn_mnist/label_noise_sweep/01_dataset/src/make_dataset.py --dataset-index 0
python 03_dnn_mnist/label_noise_sweep/03_reference_search/src/reference_search.py --dataset-index 0
python 03_dnn_mnist/label_noise_sweep/04_sampling/src/sampling.py --dataset-index 0 --shard-index 0 --shard-count 5
```

These commands are dry runs. Add `--execute` for calculation. Digit-pair sampling uses the corresponding path and `--shard-count 12`.

- **Dataset generation**
  - **Utils Dependencies**: `datasets`.
  - **Purpose**: Construct paired label-noise data or balanced digit-pair tasks.
- **Complexity**
  - **Purpose**: Calculate C_MS and validate the frozen pair selection.
- **Reference search**
  - **Utils Dependencies**: `reference_training`.
  - **Purpose**: Retain zero-error references under the fixed loss.
- **Sampling**
  - **Utils Dependencies**: `shell_smc`.
  - **Purpose**: Estimate local entropy and direct radial derivatives.
- **Summary**
  - **Purpose**: Calculate reference-then-dataset profiles and condition metrics.

Summary scripts support `--config`, `--output-dir`, `--check-only`, `--execute`, and `--force`. Input paths are relative to the experiment root; configured outputs are relative to the stage root.

## Configuration

- **default.json**: Actual numerical settings and paths.
- **objective.json**: Loss (1,.01) with shell beta=100 applied once.
- **resources.json**: At most 24 CPU threads and two GPUs; device choice is made at execution.
- **frozen_pair_manifest.json**: Selected pairs and ranks.
- **r1_weighted_accuracy.json**: Accuracy summary used by legacy sampling shards.

Existing files are reused by filename. Numerical checks cover missing/duplicate coordinates, finite values, and the required grids.

## Training Accuracy at r=1

Each experiment has `05_proxy_local_entropy/src/make_r1_accuracy.py`.

- **Utils Dependencies**: `r1_accuracy`.
- **Purpose**: Rebuild the JSON from 500 label-noise or 1,200 digit-pair reference observations.
- **Input**: `05_proxy_local_entropy/frozen_inputs/r1_accuracy_per_reference.csv`.
- **Execution**: `--execute` creates missing outputs; `--check-only` compares means/SEM; `--force` explicitly rebuilds.

No GPU or resampling is needed. If the JSON is absent, the entropy summary can aggregate the same input in memory.

New sampling records r=1 terminal accuracy directly. Fresh and archived values are not mixed within a condition. QC sentinel coordinates in legacy shards are partial snapshots, not complete particle pools.

## Results Storage

Numerical results are stored in `summarized_outputs/`; compact accuracy observations are in `05/frozen_inputs/`. Large MNIST caches, dataset NPZ files, references, and sampling shards are excluded.
