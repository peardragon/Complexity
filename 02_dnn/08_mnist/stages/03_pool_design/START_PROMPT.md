GOAL: Execute Stage 03 only: pool design and experiment contract.

Read:
- `02_dnn/08_mnist/00_GLOBAL_GOAL.md`
- `02_dnn/08_mnist/01_FORMULAS_PM_SAIS.md`
- `02_dnn/08_mnist/stages/03_pool_design/README.md`

Do not run heavy training or sampling.

Implement or specify:
1. Model spec:
   - `196-16-16-1-tanh`
   - `P=3441`
   - flatten/unflatten order
   - forward function
2. Loss:
   - CE_mean
   - CE_sum
   - gamma_ce = beta * n_train
3. Pool1:
   - optimizer-induced exact references
   - `train_error == 0`
   - reference policy ablations supported
4. Pool2:
   - PM-SAIS hard shell
   - radius grid
   - lambda candidates
   - H-ladder diagnostics
5. QC:
   - finite refs
   - ESS
   - split logZ
   - bootstrap
   - no-claim policy

Outputs:
`02_dnn/08_mnist/runs/smoke/03_pool_design/`
- `POOL_CONTRACT.md`
- `POOL_CONTRACT.json`
- `MODEL_SPEC.md`
- `QC_GATES.md`
- `REPORT.md`
- `run_config_resolved.json`
- `QC_STATUS.json`

Also create:
- `02_dnn/08_mnist/src/mnist14_model.py`
- `02_dnn/08_mnist/tests/test_stage03_model_spec.py`

Tests:
- P count equals 3441.
- flatten/unflatten roundtrip.
- CE finite.
- train error function works.

Stop after Stage 03.
