# 04 Sampling

PM-SAIS shell sampling.

- `src/sample_refpool1024_all_radii.py`: current mechanical 1024-sample production runner.
- `src/resample_mnist10_local_support.py`: reusable unit sampler and aggregation helpers.
- `raw_outputs/`: sampling runs and unit summaries.
- `figures/`: sampling figures.
- `QC/`: sampling QC diagnostics.

Main command:

```bash
python 04_sampling/src/sample_refpool1024_all_radii.py
```

Advanced 90ref production grid:

```bash
04_sampling/src/run_refpool1024_advanced_90ref.sh
```

This uses radii `0.10..2.50` in `0.05` steps and writes phi derivative tables during aggregation.
