# Digit-Pair Experiment

Natural binary classification tasks for Figs. 7–8.

## Project Overview

C_MS is calculated for all 45 pairs on ten balanced datasets each. The twelve tasks at mean-complexity ranks 1,5,9,...,45 are selected.

The selected tasks, written here from higher to lower complexity, are:

```text
4/9, 3/8, 5/9, 2/7, 4/5, 0/2,
2/9, 7/8, 4/6, 1/6, 0/9, 0/1
```

The authoritative identities and ranks are in each stage's `config/frozen_pair_manifest.json`.

## Workflow & Dependencies

- **01_dataset**
  - **Utils Dependencies**: `datasets`.
  - **Purpose**: Generate all 45 tasks and record replica-level complexity in the raw pair manifests.
- **02_complexity_measure**
  - **Purpose**: Average ten replicas, rank by C_MS, and use (digit_a,digit_b) as the deterministic tie-break.
- **03_reference_search / 04_sampling**
  - **Purpose**: Train and sample the twelve frozen tasks.
- **05_proxy_local_entropy**
  - **Purpose**: Aggregate profiles, radial variation, and r=1 accuracy.

Stage 02 owns `summarized_outputs/digit_pairwise_complexity_summary.json`. Its mean-ranking selection is checked against the frozen manifest. Some seeds are reused across tasks; no new seed scheme is substituted.

See the [MNIST README](../README.md) for execution commands.

## Results Storage

C_MS and sampling QC contain 120 rows; the energetic profiles contain 1,200 rows; condition metrics contain twelve rows. Original r=1 capture details are in [frozen_inputs/README.md](05_proxy_local_entropy/frozen_inputs/README.md).
