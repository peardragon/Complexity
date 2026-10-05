# Saved r=1 Accuracy: Synthetic

Original source: `miscellaneous/additional_experiemnts/08_synthetic_r1_weighted_accuracy_c100`.

## Workflow & Dependencies

- **replay_r1_accuracy.py**: Original terminal-particle accuracy evaluation.
- **aggregate_and_plot.py**: Original reference-then-dataset aggregation.
- **r1_accuracy_per_reference.csv**: Compact observations from 1,080 saved jobs and 10,800 references.

The CSV retains split accuracies, split-logZ mixture weights, and reference accuracies. It does not contain complete particles or training datasets.

## Aggregation

Average ten references within each dataset, then calculate the mean and sample SEM across 60 datasets. The original NumPy mean and std(ddof=1) convention is preserved. Reconstructed means and SEM match the retained JSON with maximum difference 0.0.

## Results Storage

Run `src/make_r1_accuracy.py --execute` from this stage's parent context to create the JSON. Existing outputs are skipped; `--force` explicitly rebuilds them. Run `--check-only` for a read-only comparison.

To import original replay JSON files, use `--source-dir <dir> --execute --output-dir <dir>`. This operation does not run GPU or SMC.
