# Stage 01 — Dataset prepare

## Goal

Prepare MNIST 28x28 -> 14x14 datasets with fixed input marginal and three label rules.

## Read first

1. `02_dnn/08_mnist/README.md`
2. `02_dnn/08_mnist/00_GLOBAL_GOAL.md`
3. `02_dnn/08_mnist/01_FORMULAS_PM_SAIS.md` when formulas are needed
4. this stage `START_PROMPT.md`

## Execution mode

This stage is designed for GOAL mode. The stage prompt is self-contained, but it assumes the global contract in `02_dnn/08_mnist`.

Run only this stage unless the master GOAL prompt explicitly instructs sequential execution.

## Required outputs

Each stage must produce:

```text
REPORT.md
run_config_resolved.json
QC_STATUS.json
```

in its stage run directory.

## Stop condition

If a hard QC gate fails, write `STAGE_BLOCKED.md` and stop.
