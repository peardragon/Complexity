# MASTER_ALL_STAGES_EXECUTION_PROMPT

Use this prompt when you want Codex to perform all smoke stages in one non-interactive run.

Read `02_dnn/08_mnist/README.md` and all stage README files. Execute the smoke pipeline from Stage 00 to Stage 06. Use the stage START_PROMPT files as the authoritative task definitions. Do not skip a stage. Do not promote to candidate or final.

If a stage fails QC, stop immediately and write:

```text
02_dnn/08_mnist/runs/smoke/<stage>/STAGE_BLOCKED.md
```

The blocked report must contain:

- exact failing condition;
- observed metric;
- expected threshold;
- files already created;
- recommended next safe action.

If all stages pass, write the final report and print the supported d_raw range.
