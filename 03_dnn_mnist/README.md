# 03 DNN MNIST

Stage-organized MNIST DNN local-entropy workspace, aligned with the neighboring `02_dnn` layout.

## Layout

- `01_dataset_gen/`: dataset inspection and dataset-facing figures.
- `02_complexity_measure/`: reference-family and complexity diagnostics.
- `03_reference_search/`: exact-reference search helpers and extra reference pools.
- `04_sampling/`: PM-SAIS shell sampling runners, raw unit summaries, QC tables, and sampling figures.
- `05_proxy_local_entropy/`: phi(d), proxy local entropy, derivative, and post-sampling reports.
- `shared/src_cache_mnist/`: local MNIST pipeline cache used by the stage scripts.
- `backup/20260618_001659/`: previous top-level `runs/`, `scripts/`, and source cache snapshot.

The old top-level run history was moved into backup. New outputs should go under the stage that owns them, usually `raw_outputs/`, `figures/`, or `QC/`.

## Current Sampling Entry Point

```bash
python 04_sampling/src/sample_refpool1024_all_radii.py
```

Default policy:
- 60 references per rule.
- radii `0.1..2.5` in `0.1` steps.
- `1024` samples per rule/reference/radius unit.
- no QC-gated task selection.
- `unit_summary.json` plus `samples.npz` per unit.
- CPU-limited by thread count and GPU disabled by default via `--device cpu`.
