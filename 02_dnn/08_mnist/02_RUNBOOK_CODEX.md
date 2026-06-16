# 02_RUNBOOK_CODEX

## Recommended execution modes

### Smoke

Use smoke first.

```bash
cd /path/to/Complexity
codex exec --cd /path/to/Complexity --ask-for-approval never --sandbox workspace-write - < 02_dnn/08_mnist/prompts/MASTER_GOAL_MODE_START_PROMPT.md
```

### Stage-by-stage

```bash
cd /path/to/Complexity
codex exec --cd /path/to/Complexity --ask-for-approval never --sandbox workspace-write - < 02_dnn/08_mnist/stages/01_dataset_prepare/START_PROMPT.md
```

### Resume

If Codex stops after a stage, launch the next stage prompt rather than asking it to infer the next step.

## Rule

- Stage 00 may inspect repo structure.
- Stage 01–06 must implement inside `02_dnn/08_mnist`.
- Candidate/final promotion must use Stage 07 prompts and only after smoke QC passes.

## Expected status files

Each stage must write:

```text
runs/smoke/<stage_name>/REPORT.md
runs/smoke/<stage_name>/run_config_resolved.json
runs/smoke/<stage_name>/QC_STATUS.json
```

If blocked:

```text
runs/smoke/<stage_name>/STAGE_BLOCKED.md
```

## Minimal final smoke proof

The smoke run is acceptable only if these exist:

```text
02_dnn/08_mnist/runs/smoke/final_report/phi_by_rule_radius.csv
02_dnn/08_mnist/runs/smoke/final_report/qc_pass_by_rule_radius.csv
02_dnn/08_mnist/runs/smoke/final_report/final_claim_table.csv
02_dnn/08_mnist/runs/smoke/final_report/REPORT.md
02_dnn/08_mnist/runs/smoke/final_report/figures/fig04_phi_energy_three_rules_main.png
```
