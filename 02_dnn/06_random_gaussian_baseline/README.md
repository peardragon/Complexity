# Random Gaussian Baseline

This folder keeps the Gaussian random baseline separate from the existing
`01_dataset_gen` through `05_proxy_local_entropy` outputs.

Main entry point:

```bash
python 06_random_gaussian_baseline/scripts/gaussian_baseline_pipeline.py status
```

Full detached run:

```bash
python 06_random_gaussian_baseline/scripts/gaussian_baseline_pipeline.py start-full-pipeline
```

The pipeline uses:

- 90 Gaussian datasets in one baseline tag.
- 30 exact references per dataset.
- Dense shell radii `0.01..2.50`.
- CPU affinity `0-15` and physical GPUs `2,3` by default.

The default baseline generator is
`iid_gaussian_features_balanced_random_labels_v1`: `X_raw ~ N(0, I_2)`,
feature-wise normalized `X_train`, and balanced random labels independent of
the features. A single compatibility beta tag, `0.05`, is retained only so the
existing beta/radius aggregation code can consume the baseline outputs; it is
not a data-generation parameter.

An earlier over-wide dry run used 18 synthetic beta tags. The active baseline
run is the 90-dataset run rooted at `gaussian_random_90_dataset` and
`gaussian_random_90_dataset_30_reference`.
