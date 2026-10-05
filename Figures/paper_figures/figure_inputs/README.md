# Figure Inputs

Compact inputs for the paper figure notebook.

## Contents

- Numerical summary CSV files.
- Four representative synthetic dataset NPZ files.
- The supplied Fig. 1 PNG.
- Four retained MNIST UMAP PDF panels.

## Workflow & Dependencies

`../config/input_sources.json` maps the repository sources to these staged files. The default build synchronizes changed sources and redraws affected figures. `--check-only` checks agreement without writing.

The four NPZ files are explicit Git-ignore exceptions. Full datasets, reference pools, and SMC shards are not included.
