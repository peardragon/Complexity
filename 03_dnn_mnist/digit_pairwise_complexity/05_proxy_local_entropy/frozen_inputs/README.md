# Saved r=1 Accuracy: MNIST Digit Pairs

Original source: `miscellaneous/additional_experiemnts/23_digit_pair_cms_10dataset_mean_rank_production/downstream`.

## Workflow & Dependencies

- **src/replay_r1_accuracy.py**: Original accuracy evaluation.
- **src/aggregate_r1_and_seal.py**: Original aggregation and JSON construction.
- **r1_accuracy/raw_outputs/digit_pair**: 120 saved jobs with 1,200 reference observations.
- **r1_accuracy_per_reference.csv**: Compact input used by the public generator.

The records comprise 90 replay jobs and 30 reused jobs with identical training inputs and references. The reused conditions are pair_4_9, pair_3_8, and pair_4_6. Their current r=1 logZ and direct derivatives also match the original records.

The historical pair_4_6 rank changed from 37 to 33, while the scientific arrays remained identical. The compact input uses condition/dataset/reference coordinates, not historical rank or metadata-hash gates. Earlier Study 09 digit-pair tasks are not used.

## Aggregation

Average ten references within each dataset, then calculate the mean and sample SEM across ten datasets using statistics.fmean/stdev. Reconstructed means and SEM match the retained JSON with maximum difference 0.0.

The CSV retains split/reference accuracies and mixture weights; it does not contain complete particles or MNIST images.

## Results Storage

Use `src/make_r1_accuracy.py --execute` for missing JSON outputs, `--force` to rebuild, and `--check-only` for a read-only comparison.

Original replay files can be imported with `--source-dir <dir> --execute --output-dir <dir>`. No GPU or SMC is performed.
