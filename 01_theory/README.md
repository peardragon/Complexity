# Theory: Perceptron Benchmark

Replica-symmetric calculation and finite-size shell SMC for Fig. 2, Appendix B, and Supplement S1–S8.

## Project Overview

- **01_theory_analytic**: Replica-symmetric curve at alpha=0.1.
- **02_theory_sampling**: N=40,80,160,320; ten datasets and ten references per dataset.
- **Particle control**: 2^15 particles for the main comparison, with the retained N=320 particle ladder.
- **Aggregation**: Reference mean within each dataset, then dataset mean and SEM.

## Workflow & Dependencies

```bash
python 01_theory/01_theory_analytic/src/theory_full_rs.py
python 01_theory/02_theory_sampling/src/make_summarized_outputs.py --check-only
```

- **theory_full_rs.py**
  - **Utils Dependencies**: `rs_refinement_core`.
  - **Purpose**: Calculate the analytic curve; existing results are skipped unless `--force` is used.
- **sampling.py**
  - **Utils Dependencies**: `schedule`, `smc`, `perceptron`, `input_generation`.
  - **Purpose**: Run a selected shard, generating missing required inputs first.
- **make_summarized_outputs.py**
  - **Utils Dependencies**: `aggregate`.
  - **Purpose**: Rebuild `phi_by_sampling.csv` from complete scalar shards.

For an independent aggregation with the raw shards available:

```bash
python 01_theory/02_theory_sampling/src/make_summarized_outputs.py --aggregate --output-dir /path/to/comparison
```

## Input Requirements

The original pool recipes are recovered from the pre-revision source. Existing NPZ files are reused without replacement. Missing inputs are written to `raw_outputs/dataset_pool/N_*/dataset_*/dataset.npz` and `raw_outputs/reference_pool/N_*/dataset_*/ref_*/reference.npz`.

```bash
python 01_theory/02_theory_sampling/src/make_datasets.py --execute
python 01_theory/02_theory_sampling/src/make_references.py --execute
```

Without `--execute`, the commands report the existing/missing files. `--check-only` requires the selected files to exist without writing. `--n-values` and `--dataset-index` select a subset; `--output-root` supports a separate comparison directory.

Dataset and reference seeds, reference burn/thin, and numerical margin epsilon are read from `config/default.json/input_generation`. The reference target is exp(-||theta||²/2) times the hard-feasibility indicator, not the DNN training objective. New reference draws need not be identical to the original saved vectors.

Settings are stored in each `config/default.json`. Executable scripts are in `src/`; numerical helpers are in `src/utils/`.

## Results Storage

RS and SMC summaries are included in `summarized_outputs/`. Dataset/reference pools and sampling shards are excluded.

The full-shell term is reconstructed using `(N-2)/N`. Legacy full-shell auxiliary fields in old raw files are not used for aggregation.
