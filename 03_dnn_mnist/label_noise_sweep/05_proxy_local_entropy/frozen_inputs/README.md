# Saved r=1 Accuracy: MNIST Label Noise

Original source: `miscellaneous/additional_experiemnts/09_mnist_r1_weighted_accuracy_c100/raw_outputs/label_noise`.

## Workflow & Dependencies

- **replay_one_root.py**: Original terminal-particle accuracy evaluation.
- **aggregate_and_plot.py**: Original dataset-first aggregation.
- **r1_accuracy_per_reference.csv**: Compact observations from 50 saved jobs and 500 references.

Only label-noise records are included. The earlier digit-pair tasks from Study 09 are not used.

## Aggregation

Average ten references within each dataset, then calculate the mean and sample SEM across ten datasets using the original statistics.fmean/stdev convention. The 100 references are not treated as independent uncertainty units.

Reconstructed means and SEM match the retained JSON with maximum difference 0.0. The CSV contains split/reference accuracies and mixture weights, not complete particles or MNIST images.

## Results Storage

Use `src/make_r1_accuracy.py --execute` to create missing JSON outputs, `--force` to rebuild, and `--check-only` for a read-only comparison.

Original replay files can be imported with `--source-dir <dir> --execute --output-dir <dir>`. No GPU or resampling is performed.
