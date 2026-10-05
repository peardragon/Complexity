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
  - **Utils Dependencies**: `schedule`, `smc`, `perceptron`.
  - **Purpose**: Run a selected sampling shard from the fixed input pools.
- **make_summarized_outputs.py**
  - **Utils Dependencies**: `aggregate`.
  - **Purpose**: Rebuild `phi_by_sampling.csv` from complete scalar shards.

For an independent aggregation with the raw shards available:

```bash
python 01_theory/02_theory_sampling/src/make_summarized_outputs.py --aggregate --output-dir /path/to/comparison
```

## Input Requirements

The current `make_datasets.py` and `make_references.py` validate the original 40 datasets and 400 references. They are not pool generators. New sampling requires these inputs to be supplied separately.

Settings are stored in each `config/default.json`. Executable scripts are in `src/`; numerical helpers are in `src/utils/`.

## Results Storage

RS and SMC summaries are included in `summarized_outputs/`. Dataset/reference pools and sampling shards are excluded.

The full-shell term is reconstructed using `(N-2)/N`. Legacy full-shell auxiliary fields in old raw files are not used for aggregation.
