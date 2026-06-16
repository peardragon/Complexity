# 02_dnn/08_mnist — MNIST14 GOAL-mode PM-SAIS pipeline

## Purpose

This directory is a self-contained instruction and runbook layer for the MNIST-14x14 real-data extension.

The execution model is:

```text
chat prompt = short GOAL
Codex reads = 02_dnn/08_mnist/README.md + stage README/START_PROMPT files
Codex executes = stages in order, with smoke -> candidate -> final gates
```

The scientific goal is:

\[
\boxed{\text{Same MNIST input marginal, different label rules, same PM-SAIS estimator, compare }\phi(d).}
\]

This means:

1. Preserve recognizable MNIST image morphology by using 28x28 -> 14x14 average pooling, not PCA2.
2. Use one small fully-connected 3NN architecture whose parameter dimension remains sampling-feasible.
3. Fix the input marginal and vary only the label rule:
   - `real_even_odd`
   - `teacher_nn`
   - `random_label`
4. Construct Pool 1 as optimizer-induced exact references.
5. Construct Pool 2 by PM-SAIS on hard raw-distance shells.
6. Report \(\Delta\phi_{\rm full}(d)\) and especially \(\Delta\phi_{\rm energy}(d)\), with strict QC and no-claim failed radii.

## Directory map

```text
02_dnn/08_mnist/
  README.md
  AGENTS.md
  00_GLOBAL_GOAL.md
  01_FORMULAS_PM_SAIS.md
  02_RUNBOOK_CODEX.md
  03_OUTPUT_CONTRACT.md
  04_QC_GATES.md
  05_CLAIM_POLICY.md
  MANIFEST.md

  prompts/
    MASTER_GOAL_MODE_START_PROMPT.md
    MASTER_ALL_STAGES_EXECUTION_PROMPT.md
    CANDIDATE_PROMOTION_PROMPT.md
    FINAL_PROMOTION_PROMPT.md

  stages/
    00_repo_audit/
      README.md
      START_PROMPT.md
      CHECKLIST.md
    01_dataset_prepare/
      README.md
      START_PROMPT.md
      CHECKLIST.md
    02_complexity_measure/
      README.md
      START_PROMPT.md
      CHECKLIST.md
    03_pool_design/
      README.md
      START_PROMPT.md
      CHECKLIST.md
    04_exact_reference_search/
      README.md
      START_PROMPT.md
      CHECKLIST.md
    05_pool2_pm_sais_sampling/
      README.md
      START_PROMPT.md
      CHECKLIST.md
    06_results_figures/
      README.md
      START_PROMPT.md
      CHECKLIST.md
    07_candidate_final_promotion/
      README.md
      START_PROMPT.md
      CHECKLIST.md

  templates/
    config_smoke.yaml
    config_candidate.yaml
    config_final.yaml

  scripts/
    run_codex_stage.sh
    run_goal_mode.sh
```

## How to start in GOAL mode

From the repository root, paste this to Codex:

```text
GOAL: Execute the MNIST14 PM-SAIS experiment under 02_dnn/08_mnist.
First read 02_dnn/08_mnist/README.md, 00_GLOBAL_GOAL.md, and all stage README.md files.
Then execute stages 00 through 06 in order at smoke scale.
Do not modify old retained production outputs.
Stop on any QC failure and write STAGE_BLOCKED.md with exact reason and next safe action.
```

A longer prompt is available at:

```text
02_dnn/08_mnist/prompts/MASTER_GOAL_MODE_START_PROMPT.md
```

## Stage order

| Stage | Directory | Goal |
|---:|---|---|
| 00 | `stages/00_repo_audit` | Audit repo conventions and create local 08_mnist skeleton without touching retained outputs. |
| 01 | `stages/01_dataset_prepare` | Prepare MNIST 28x28 -> 14x14 datasets for three label rules. |
| 02 | `stages/02_complexity_measure` | Compute graph TV/NMSTV-style label complexity for each dataset. |
| 03 | `stages/03_pool_design` | Freeze model, loss, Pool1, Pool2, \(\phi(d)\), QC, and run scales. |
| 04 | `stages/04_exact_reference_search` | Train exact references for 196-16-16-1 tanh network. |
| 05 | `stages/05_pool2_pm_sais_sampling` | Run PM-SAIS / optional PM-RLI H-ladder on hard shells. |
| 06 | `stages/06_results_figures` | Aggregate \(\phi(d)\), QC, figures, and report. |
| 07 | `stages/07_candidate_final_promotion` | Scale smoke -> candidate -> final only if QC supports it. |

## Non-negotiable constraints

- Work under `02_dnn/08_mnist` unless explicitly reading reusable code from prior stages.
- Do not overwrite or delete existing `02_dnn/01_*` to `02_dnn/07_*` retained outputs.
- Do not use generated images. Figures must be computed from actual data.
- Do not claim beyond QC-passed support.
- Do not claim the reference ensemble is an exact sample from \(P_{\rm ref}^0\); it is optimizer-induced unless a target-aware reference sampler is implemented.
- Keep all smoke/candidate/final outputs separate.
- Every stage must write:
  - `REPORT.md`
  - `run_config_resolved.json`
  - machine-readable CSV/JSON summaries
  - `QC_STATUS.json`
- Failed radius/stage policy: `no_claim` or `STAGE_BLOCKED.md`, never silent success.

## Main deliverable

The final smoke deliverable is:

```text
02_dnn/08_mnist/runs/smoke/mnist14_3rule_1024_5_reference/
  final_report/REPORT.md
  final_report/final_claim_table.csv
  final_report/phi_by_rule_radius.csv
  final_report/qc_pass_by_rule_radius.csv
  figures/fig04_phi_energy_three_rules_main.png
```

Candidate/final promotion is handled only after smoke QC passes.
