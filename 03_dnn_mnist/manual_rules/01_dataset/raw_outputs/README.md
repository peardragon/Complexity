# raw_outputs

Raw dataset artifacts live here and are not tracked by git.

Expected standalone layout:

- `original_mnist/mnist_openml_uint8.npz`: stage-local OpenML MNIST cache with `X` and `y`.
- `rule_*/dataset.npz`: generated MNIST10 manual-rule payloads.
- `rule_*/dataset_meta.json`: local validation metadata.
- `rule_*/generation_meta.json`: source MNIST, split, and rule-generation provenance.

Run `python src/make_dataset.py --source-cache /path/to/mnist_openml_uint8.npz --no-download` from this stage's parent repo to seed `original_mnist/` from an existing cache, or omit `--no-download` to allow OpenML download when the cache is missing.
