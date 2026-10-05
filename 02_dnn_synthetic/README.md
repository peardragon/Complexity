# Synthetic DNN: Controlled Label Organization

Synthetic experiment for Figs. 3–4 and Appendices A/C.

## Project Overview

- **Data**: Beta=0.05,0.07,...,0.39; 60 datasets per condition and ten references per dataset.
- **Model**: 2–48–48–1 tanh network, P=2545.
- **Sampling**: r=0.01–2.50; two independent pools of 512 particles.
- **Outputs**: C_MS, local-entropy profiles, direct radial derivatives, radial variation, and r=1 accuracy.

## Directory Structure

```text
02_dnn_synthetic/
├── 01_dataset/                 # Mutual-kNN graph and Kawasaki labels
├── 02_complexity_measure/      # C_MS
├── 03_reference_search/        # Zero-error reference search
├── 04_sampling/                # Shell SMC and numerical QC
└── 05_proxy_local_entropy/     # Profiles, condition metrics, r=1 captures
```

Each stage contains `config/`, `src/`, and `summarized_outputs/` where applicable.

## Workflow & Dependencies

```bash
python 02_dnn_synthetic/01_dataset/src/make_dataset.py --start 0 --stop 1
python 02_dnn_synthetic/03_reference_search/src/reference_search.py --start 0 --stop 1
python 02_dnn_synthetic/04_sampling/src/sampling.py --dataset-job 0 --shell-pass odd
```

The commands above are dry runs. Add `--execute` for calculation.

- **Dataset generation**
  - **Utils Dependencies**: `dataset`.
  - **Purpose**: Build the mutual-kNN graph, update balanced labels, and standardize inputs.
- **Reference search**
  - **Utils Dependencies**: `reference`, `model`, `objective`.
  - **Purpose**: Retain ten distinct zero-training-error endpoints.
- **Sampling**
  - **Utils Dependencies**: `smc`, `radial`, `vmf`.
  - **Purpose**: Estimate angular logZ and direct radial scores.
- **Summary**
  - **Purpose**: Aggregate references before calculating dataset means and SEM.

Summary scripts support `--config`, `--output-dir`, `--raw-root`, and `--check-only`. `--execute` writes missing outputs; `--force` explicitly rebuilds them.

## Configuration

- **default.json**: Numerical conditions, dimensions, grids, and paths.
- **objective.json**: Base loss coefficients (1,.01), scaled by shell beta=100 to (100,1).
- **resources.json**: At most 24 CPU threads and two GPUs; select devices with `CUDA_VISIBLE_DEVICES` or `--device`.

Code/environment hashes are metadata, not reuse gates.

## Training Accuracy at r=1

```bash
python 02_dnn_synthetic/05_proxy_local_entropy/src/make_r1_accuracy.py --check-only
python 02_dnn_synthetic/05_proxy_local_entropy/src/make_r1_accuracy.py --execute
```

The generator reaggregates 10,800 saved reference observations in `05/frozen_inputs/` without GPU or SMC. Original sources and aggregation details are in [the capture README](05_proxy_local_entropy/frozen_inputs/README.md).

New sampling records terminal accuracy directly. Legacy summaries use the preserved JSON, or the capture table if the JSON is missing. Partial fresh/archive mixtures are rejected.

## Results Storage

Compact results are stored in each stage's `summarized_outputs/`. Large datasets, reference packs, and sampling shards are excluded from the release.
