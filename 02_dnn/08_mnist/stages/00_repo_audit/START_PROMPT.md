You are working in the repository root.

GOAL: Execute Stage 00 only: repo audit and `02_dnn/08_mnist` skeleton.

Read:
- `README.md`
- `AGENTS.md` if present
- `02_dnn/README.md` if present
- `02_dnn/08_mnist/README.md`
- `02_dnn/08_mnist/00_GLOBAL_GOAL.md`
- this stage README

Tasks:
1. Inspect existing DNN stage layout and reusable code under `02_dnn/01_*` to `02_dnn/07_*`.
2. Do not modify any retained production output.
3. Create missing local directories under `02_dnn/08_mnist`:
   - `src/`
   - `config/`
   - `tests/`
   - `runs/smoke/`
   - `runs/candidate/`
   - `runs/final/`
4. Write audit outputs:
   - `02_dnn/08_mnist/runs/smoke/00_repo_audit/AUDIT_REPORT.md`
   - `02_dnn/08_mnist/runs/smoke/00_repo_audit/REUSE_MAP.md`
   - `02_dnn/08_mnist/runs/smoke/00_repo_audit/DIRECTORY_TREE.md`
   - `02_dnn/08_mnist/runs/smoke/00_repo_audit/QC_STATUS.json`
   - `02_dnn/08_mnist/runs/smoke/00_repo_audit/run_config_resolved.json`
5. Identify code that should be reused or copied:
   - dataset utilities
   - model flatten/unflatten conventions
   - PM-SAIS/vMF/logM utilities
   - proxy aggregation / plotting conventions
6. Stop after Stage 00.

Acceptance:
- No retained output changed.
- The audit files exist.
- `QC_STATUS.json` has `status: "pass"` or `status: "blocked"`.

Print the next prompt path:
`02_dnn/08_mnist/stages/01_dataset_prepare/START_PROMPT.md`
