GOAL: Execute Stage 01 only: MNIST14 dataset preparation.

Read:
- `02_dnn/08_mnist/README.md`
- `02_dnn/08_mnist/00_GLOBAL_GOAL.md`
- `02_dnn/08_mnist/stages/01_dataset_prepare/README.md`

Implement under:
`02_dnn/08_mnist/src/`

Smoke settings:
- experiment_id: `mnist14_3rule_1024`
- n_splits: 3
- n_train: 256
- n_test: 2048
- labels: `real_even_odd`, `teacher_nn`, `random_label`
- input: MNIST 28x28 -> 14x14 by exact 2x2 average pooling
- flatten to 196
- standardize by train mean/std per split

MNIST loading:
- Prefer local `data/mnist` or torchvision if available.
- Do not silently download in production/final.
- For smoke, downloading is allowed only if explicitly configured and recorded.

Required arrays in each dataset NPZ:
- `X_train`: `(n_train,196)`, float32, standardized
- `y_train`: `(n_train,)`, int8, values {-1,+1}
- `X_test`: `(n_test,196)`, float32
- `y_test`: `(n_test,)`, int8
- `X_train_raw14`: `(n_train,196)`, float32, unstandardized
- `X_test_raw14`: `(n_test,196)`, float32
- `digit_train`
- `digit_test`

Label rules:
1. `real_even_odd`: +1 even digit, -1 odd digit.
2. `teacher_nn`: frozen random tanh teacher `196->32->32->1`; threshold by train median logit.
3. `random_label`: balanced iid random train labels; independently generated test labels; metadata says test accuracy is not generalization.

Outputs:
`02_dnn/08_mnist/runs/smoke/01_dataset_prepare/`
- `dataset_index.csv`
- `raw_datasets/split_000/<rule>/dataset.npz`
- `metadata/split_summary.csv`
- `metadata/label_balance_summary.csv`
- `figures/fig01_mnist_28_vs_14_montage.png`
- `figures/fig02_label_balance_by_rule.png`
- `REPORT.md`
- `run_config_resolved.json`
- `QC_STATUS.json`

Tests:
Create and run:
`02_dnn/08_mnist/tests/test_stage01_dataset_prepare.py`

QC:
- shapes correct;
- labels finite and in {-1,+1};
- train label balance 0.45 to 0.55 for all rules, unless tie/warning documented;
- no NaN/Inf;
- montage exists.

Stop after Stage 01.
