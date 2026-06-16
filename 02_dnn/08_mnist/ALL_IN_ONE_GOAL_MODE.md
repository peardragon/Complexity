# MNIST14 GOAL-mode all-in-one package



---

## FILE: README.md

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


---

## FILE: 00_GLOBAL_GOAL.md

# 00_GLOBAL_GOAL — MNIST14 PM-SAIS research contract

## One-sentence goal

\[
\boxed{
\text{Use one real MNIST input marginal, vary only the label rule, and compare reference-conditioned shell free entropy } \phi(d).
}
\]

## Why this experiment exists

The previous 2D synthetic pipeline established a PM-SAIS/RLI route for finite 3NN shell entropy. The MNIST14 extension is a closure experiment:

- input is now real image data, not synthetic 2D fields;
- morphology is preserved by 14x14 average pooling;
- architecture is redesigned but kept small enough for shell sampling;
- the estimator is fixed before looking at results.

## Main scientific design

### Data

\[
28\times28 \rightarrow 14\times14 \rightarrow x\in\mathbb R^{196}.
\]

Use exact 2x2 average pooling. Store both raw14 and standardized vectors.

### Label rules

All regimes share the same input marginal.

| rule | definition | interpretation |
|---|---|---|
| `real_even_odd` | \(y=+1\) for even digit, \(-1\) for odd digit | semantic structured label |
| `teacher_nn` | \(y=\operatorname{sign}(T(x)-\operatorname{median}_{train}T)\) | architecture-compatible synthetic rule |
| `random_label` | iid balanced random \(y\in\{-1,+1\}\) | no label structure / memorization control |

### Model

Main architecture:

\[
196\rightarrow16\rightarrow16\rightarrow1,\quad \tanh.
\]

Parameter count:

\[
P=196\cdot16+16+16\cdot16+16+16+1=3441.
\]

Backup architecture only if random-label exact reference rate fails badly:

\[
196\rightarrow24\rightarrow24\rightarrow1,\quad P=5353.
\]

### Loss convention

Labels \(y_i\in\{-1,+1\}\), logit \(f_\theta(x_i)\).

\[
\ell_i(\theta)=\log(1+\exp[-y_i f_\theta(x_i)]).
\]

\[
CE_{\rm mean}=\frac1n\sum_i\ell_i,\qquad CE_{\rm sum}=n\,CE_{\rm mean}.
\]

Sampling target:

\[
U(\theta)=\beta CE_{\rm sum}(\theta)+\lambda_{\rm reg}\frac{\|\theta\|^2}{2P}.
\]

If code returns `CE_mean`, use:

\[
\gamma_{\rm CE}=\beta n_{\rm train}
\]

and compute residual weights with \(\exp[-\gamma_{\rm CE}CE_{\rm mean}]\).

Default:

\[
\beta=1,\qquad \lambda_{\rm reg}\in\{1,10,50,100\}\text{ selected by pilot.}
\]

### Pool 1

Pool 1 is the reference ensemble.

Practical reference law:

\[
\boxed{\text{optimizer-induced exact reference ensemble}}
\]

Acceptance:

\[
\mathrm{train\ error}(\tilde\theta)=0.
\]

Record reference-bias diagnostics:

- `exact_opt_unweighted`
- `exact_opt_L2_reweighted`
- `norm_matched_exact`

Do not claim exact sampling from:

\[
P_{\rm ref}^{0}(\theta\mid D)
\propto
\mathbf 1\{\mathrm{err}=0\}
e^{-\lambda_{\rm ref}\|\theta\|^2/(2P)}
\]

unless a target-aware reference sampler is implemented.

### Pool 2

Pool 2 is the reference-conditioned hard shell.

\[
d=d_{\rm raw}=\frac{\|\theta-\tilde\theta\|}{\sqrt P}.
\]

\[
\theta(d,u)=\tilde\theta+\sqrt P\,d\,u,\qquad \|u\|=1.
\]

Main estimator: PM-SAIS \(H=\infty\). Optional PM-RLI \(H\in\{8,4,2\}\) diagnostic.

### Main observables

\[
Z(d)=e^{-\lambda d^2/2}M_P(\kappa_d)
\mathbb E_{u\sim{\rm vMF}}
[
e^{-\gamma CE_{\rm mean}(\theta(d,u))}
].
\]

\[
\Delta\phi_{\rm full}(d)
=
\frac{P-1}{P}\log\frac d{d_0}
+
\frac1P[\log Z(d)-\log Z(d_0)].
\]

\[
\Delta\phi_{\rm energy}(d)
=
\frac1P[\log Z(d)-\log Z(d_0)].
\]

Main figure: three curves of \(\Delta\phi_{\rm energy}(d)\) for `real_even_odd`, `teacher_nn`, `random_label`.

## Claim policy

Allowed:

\[
\boxed{
\text{Same input marginal, different label rules, different PM-SAIS shell free-entropy profiles.}
}
\]

Not allowed:

\[
\boxed{
\text{We invented local entropy.}
}
\]

\[
\boxed{
\text{Optimizer-found references are exact }P_{\rm ref}^{0}\text{ samples.}
}
\]

\[
\boxed{
\text{Any radius beyond QC pass is supported.}
}
\]


---

## FILE: 01_FORMULAS_PM_SAIS.md

# 01_FORMULAS_PM_SAIS

## PM-SAIS identity

General shell partition:

\[
\Omega(s)=A(s)\mathbb E_{u\sim p_s}[F_s(u)].
\]

PM-SAIS estimates the same target using a proposal \(q_s\):

\[
\mathbb E_{p_s}[F_s(u)]
=
\mathbb E_{q_s}\left[F_s(u)\frac{p_s(u)}{q_s(u)}\right].
\]

For this project:

\[
s=d=d_{\rm raw},\qquad p_s={\rm Haar}(S^{P-1}).
\]

## Hard shell map

\[
\theta(d,u)=\tilde\theta+\sqrt P\,d\,u,\qquad \|u\|=1.
\]

## Target

\[
U(\theta)=\gamma CE_{\rm mean}(\theta;D)+\lambda\frac{\|\theta\|^2}{2P}.
\]

With extensive CE:

\[
\gamma=\beta n_{\rm train}.
\]

## L2 decomposition

\[
\|\theta(d,u)\|^2
=
\|\tilde\theta\|^2
+
Pd^2
+
2\sqrt P\,d\,\|\tilde\theta\|(\hat{\tilde\theta}\cdot u).
\]

Let:

\[
\mu_d=-\hat{\tilde\theta},\qquad
\kappa_d=\lambda d\frac{\|\tilde\theta\|}{\sqrt P}.
\]

Then the L2 angular tilt is matched by:

\[
q_d(u)={\rm vMF}(\mu_d,\kappa_d).
\]

## PM-SAIS angular partition

Dropping the reference-constant factor that cancels in relative curves:

\[
Z_{\rm PM-SAIS}(d)
=
e^{-\lambda d^2/2}M_P(\kappa_d)
\mathbb E_{u\sim q_d}
[
e^{-\gamma CE_{\rm mean}(\theta(d,u);D)}
].
\]

## Monte Carlo estimator

\[
\widehat Z(d)
=
e^{-\lambda d^2/2}M_P(\kappa_d)
\frac1m\sum_{a=1}^{m}
\exp[-\gamma CE_{\rm mean}(\theta(d,u_a);D)].
\]

Use logsumexp:

\[
\log\widehat Z(d)
=
-\lambda d^2/2+\log M_P(\kappa_d)
+\operatorname{logmeanexp}_a[-\gamma CE_a].
\]

## PM-RLI optional H-gate

\[
h(\theta)=\sqrt{2\max(CE_{\rm mean}(\theta)-CE_{\rm mean}(\tilde\theta),0)}.
\]

\[
Z_H(d)
=
e^{-\lambda d^2/2}M_P(\kappa_d)
\mathbb E_{u\sim q_d}
[
e^{-\gamma CE_{\rm mean}(\theta(d,u))}
\mathbf 1\{h(\theta(d,u))\le H\}
].
\]

\[
R_H(d)=\frac{Z_H(d)}{Z_\infty(d)}.
\]

Interpretation:

| layer | role |
|---|---|
| \(H=\infty\) | main PM-SAIS/global shell |
| \(H=8\) | broad loss-response diagnostic |
| \(H=4\) | medium sector diagnostic |
| \(H=2\) | strict local low-loss diagnostic only |

## Full and energy-only free entropy

\[
\Delta\phi_{\rm full}(d;H)
=
\frac{P-1}{P}\log\frac d{d_0}
+
\frac1P[\log Z_H(d)-\log Z_H(d_0)].
\]

\[
\Delta\phi_{\rm energy}(d;H)
=
\frac1P[\log Z_H(d)-\log Z_H(d_0)].
\]

Use \(\Delta\phi_{\rm energy}\) as the main landscape-quality comparison. Show \(\Delta\phi_{\rm full}\) with area decomposition.


---

## FILE: 02_RUNBOOK_CODEX.md

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


---

## FILE: 03_OUTPUT_CONTRACT.md

# 03_OUTPUT_CONTRACT

## Global naming

Experiment ID:

```text
mnist14_3rule_1024
```

Rules:

```text
real_even_odd
teacher_nn
random_label
```

Main architecture:

```text
196-16-16-1-tanh
P=3441
```

## Smoke paths

```text
02_dnn/08_mnist/runs/smoke/00_repo_audit/
02_dnn/08_mnist/runs/smoke/01_dataset_prepare/
02_dnn/08_mnist/runs/smoke/02_complexity_measure/
02_dnn/08_mnist/runs/smoke/03_pool_design/
02_dnn/08_mnist/runs/smoke/04_exact_reference_search/
02_dnn/08_mnist/runs/smoke/05_pool2_pm_sais_sampling/
02_dnn/08_mnist/runs/smoke/06_results_figures/
02_dnn/08_mnist/runs/smoke/final_report/
```

## Required stage outputs

### Stage 01 dataset

```text
dataset_index.csv
raw_datasets/split_000/<rule>/dataset.npz
metadata/split_summary.csv
metadata/label_balance_summary.csv
figures/fig01_mnist_28_vs_14_montage.png
```

### Stage 02 complexity

```text
complexity_by_dataset.csv
complexity_by_rule_summary.csv
graph_stats_by_dataset_k.csv
figures/fig01_nmstv_by_rule_boxplot.png
```

### Stage 04 references

```text
reference_index.csv
selected_reference_pool/split_000/<rule>/ref_000/theta.npy
selected_reference_pool/split_000/<rule>/ref_000/ref_summary.json
attempt_logs/attempts.csv
```

### Stage 05 PM-SAIS

```text
selected_lambda.json
lambda_selection_report.csv
shell_summary_by_unit.csv
shell_summary_by_rule_radius.csv
qc_by_rule_radius.csv
```

### Stage 06 final figures

```text
phi_by_rule_radius.csv
phi_bootstrap_by_rule_radius.csv
qc_pass_by_rule_radius.csv
complexity_reference_sampling_joined.csv
final_claim_table.csv
figures/fig04_phi_energy_three_rules_main.png
figures/fig05_phi_full_three_rules.png
figures/fig07_sampling_qc_pass_heatmap.png
REPORT.md
```


---

## FILE: 04_QC_GATES.md

# 04_QC_GATES

## Global QC

- No NaN/Inf in datasets, references, shell summaries, or final aggregation.
- All random seeds must be recorded.
- Failed radius policy: `no_claim`.
- Failed stage policy: write `STAGE_BLOCKED.md`.

## Dataset QC

- `X_train` shape: `(n_train, 196)`.
- `X_test` shape: `(n_test, 196)`.
- labels are int values in `{-1,+1}`.
- train class balance by rule must be between 0.45 and 0.55 for smoke unless unavoidable ties are explained.
- `teacher_nn` threshold is train median logit.
- montage figure exists and visibly preserves digit shapes.

## Complexity QC

- kNN graph edge count > 0 for all k.
- all TV/NMSTV values finite.
- random-label NMSTV should be near random baseline after normalization; warn if not.
- do not fail solely because teacher/real ordering is unexpected.

## Reference QC

- selected references have `train_error == 0`.
- theta vector length equals P.
- pairwise duplicate threshold: selected theta L2 distance > 1e-6.
- per dataset smoke target: 5 exact references.
- if random_label exact success is insufficient, first increase attempts; only then consider backup architecture.

## PM-SAIS QC

Smoke thresholds:

| field | threshold |
|---|---:|
| finite unit fraction per rule/radius | >= 0.90 |
| q05 ESS fraction | >= 0.02 |
| max split logZ/P diff | <= 0.004 |
| bootstrap sd phi | <= 0.012 |
| CESS if SMC used | >= 0.60 |

Candidate/final thresholds:

| field | threshold |
|---|---:|
| finite unit fraction per rule/radius | >= 0.95 |
| q05 ESS fraction | >= 0.04 |
| max split logZ/P diff | <= 0.004 |
| bootstrap sd phi | <= 0.012 |

## Final claim QC

A radius is claimable only if all three rules pass QC at that radius and the reference count is adequate.

Report supported range explicitly:

```text
Supported d_raw radii: [...]
No-claim d_raw radii: [...]
```


---

## FILE: 05_CLAIM_POLICY.md

# 05_CLAIM_POLICY

## Main claim

Allowed:

```text
Using the same MNIST-14x14 input marginal and the same PM-SAIS estimator, changing only the label rule produces different reference-conditioned shell free-entropy profiles.
```

## Required caveats

- References are optimizer-induced exact references unless a target-aware reference sampler is implemented.
- Random-label test accuracy is not a generalization metric.
- Full phi includes high-dimensional shell-area dominance.
- Energy-only phi is the main landscape-quality comparison.
- Any failed radius is no-claim.
- Optional \(H\le2\) is strict local diagnostic, not broad raw-shell production target.

## Forbidden claims

- "We invented local entropy."
- "HMC/weight-space sampling itself is the novelty."
- "Optimizer references are exact \(P_{
m ref}^0\) samples."
- "The result holds outside QC-passed support."
- "Random labels generalize."


---

## FILE: stages/00_repo_audit/README.md

# Stage 00 — Repo audit and 08_mnist skeleton

## Goal

Audit existing repo conventions and create a self-contained `02_dnn/08_mnist` implementation skeleton without modifying retained outputs.

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


---

## FILE: stages/00_repo_audit/START_PROMPT.md

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


---

## FILE: stages/00_repo_audit/CHECKLIST.md

# CHECKLIST — 00_repo_audit


- [ ] Existing repo conventions inspected.
- [ ] No retained production outputs modified.
- [ ] Local `02_dnn/08_mnist` skeleton exists.
- [ ] `AUDIT_REPORT.md` exists.
- [ ] `REUSE_MAP.md` exists.
- [ ] `DIRECTORY_TREE.md` exists.
- [ ] `QC_STATUS.json` exists.
- [ ] Next stage prompt printed.


---

## FILE: stages/01_dataset_prepare/README.md

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


---

## FILE: stages/01_dataset_prepare/START_PROMPT.md

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


---

## FILE: stages/01_dataset_prepare/CHECKLIST.md

# CHECKLIST — 01_dataset_prepare


- [ ] `dataset_index.csv` has 9 rows.
- [ ] Every NPZ exists.
- [ ] Every NPZ has required arrays.
- [ ] Shapes and dtypes pass.
- [ ] Label balances pass or warning is documented.
- [ ] 28x28 vs 14x14 montage exists.
- [ ] Pytest passes.
- [ ] `REPORT.md` and `QC_STATUS.json` exist.


---

## FILE: stages/02_complexity_measure/README.md

# Stage 02 — Complexity measure calc

## Goal

Compute graph TV/NMSTV-style label complexity for the three MNIST label rules.

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


---

## FILE: stages/02_complexity_measure/START_PROMPT.md

GOAL: Execute Stage 02 only: complexity measure calculation.

Read:
- `02_dnn/08_mnist/stages/02_complexity_measure/README.md`
- Stage 01 output `dataset_index.csv`

Input:
`02_dnn/08_mnist/runs/smoke/01_dataset_prepare/dataset_index.csv`

Compute graph total variation / NMSTV style complexity on standardized `X_train`.

Graph:
- multiscale kNN with k = [8,16,32]
- Euclidean distance
- symmetric graph by max weight
- weight: `exp(-dist^2/(2*sigma_k^2))`
- `sigma_k`: median nonzero kNN distance

Label TV:
\[
TV_k = \sum_{edges} w_{ij} 1[y_i\ne y_j] / \sum_{edges} w_{ij}.
\]

Balance baseline:
\[
baseline = 2p(1-p),\quad p=P(y=+1).
\]

\[
NMSTV_k = TV_k / max(baseline, 1e-12).
\]

Outputs:
`02_dnn/08_mnist/runs/smoke/02_complexity_measure/`
- `complexity_by_dataset.csv`
- `complexity_by_rule_summary.csv`
- `graph_stats_by_dataset_k.csv`
- `figures/fig01_nmstv_by_rule_boxplot.png`
- `figures/fig02_tv_by_k_rule.png`
- `figures/fig03_complexity_vs_label_balance.png`
- `REPORT.md`
- `run_config_resolved.json`
- `QC_STATUS.json`

Tests:
Create and run:
`02_dnn/08_mnist/tests/test_stage02_complexity_measure.py`

QC:
- all datasets have finite complexity;
- graph edge counts > 0;
- all three rules present;
- unexpected ordering is a warning, not a failure.

Stop after Stage 02.


---

## FILE: stages/02_complexity_measure/CHECKLIST.md

# CHECKLIST — 02_complexity_measure


- [ ] 9 dataset-level complexity rows.
- [ ] 3 rule summary rows.
- [ ] kNN graphs finite.
- [ ] TV/NMSTV finite.
- [ ] Figures exist.
- [ ] Pytest passes.
- [ ] `REPORT.md` and `QC_STATUS.json` exist.


---

## FILE: stages/03_pool_design/README.md

# Stage 03 — Pool design

## Goal

Freeze model, loss, Pool1, Pool2, PM-SAIS formulas, radius grid, and QC contract.

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


---

## FILE: stages/03_pool_design/START_PROMPT.md

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


---

## FILE: stages/03_pool_design/CHECKLIST.md

# CHECKLIST — 03_pool_design


- [ ] `POOL_CONTRACT.json` exists.
- [ ] `MODEL_SPEC.md` exists.
- [ ] `P=3441` verified.
- [ ] Model utility tests pass.
- [ ] Loss convention explicitly uses extensive CE in sampling.
- [ ] Reference law caveat included.
- [ ] `REPORT.md` and `QC_STATUS.json` exist.


---

## FILE: stages/04_exact_reference_search/README.md

# Stage 04 — Exact reference search

## Goal

Train exact references for each MNIST14 dataset/rule using the fixed small 3NN.

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


---

## FILE: stages/04_exact_reference_search/START_PROMPT.md

GOAL: Execute Stage 04 only: exact reference search.

Read:
- `02_dnn/08_mnist/runs/smoke/03_pool_design/POOL_CONTRACT.json`
- `02_dnn/08_mnist/stages/04_exact_reference_search/README.md`

Inputs:
- Stage 01 dataset index.
- Stage 03 model spec.

Smoke:
- selected_refs_per_dataset: 5
- max_attempts_per_dataset: 60
- optimizer: Adam warmup -> LBFGS polish
- full-batch preferred for reproducibility
- deterministic seeds

Reference acceptance:
\[
train\_error(	ilde	heta)=0.
\]

Store:
- theta vector
- CE_mean_train
- CE_sum_train
- CE_mean_test
- train/test error
- theta norm/norm_sq
- margin stats
- optimizer seed/attempt id
- selected policy

Outputs:
`02_dnn/08_mnist/runs/smoke/04_exact_reference_search/`
- `reference_index.csv`
- `selected_reference_pool/split_000/<rule>/ref_000/theta.npy`
- `selected_reference_pool/split_000/<rule>/ref_000/ref_summary.json`
- `attempt_logs/attempts.csv`
- `figures/fig01_reference_success_rate_by_rule.png`
- `figures/fig02_ref_ce_norm_scatter.png`
- `figures/fig03_margin_distribution_by_rule.png`
- `REPORT.md`
- `run_config_resolved.json`
- `QC_STATUS.json`

Tests:
Create and run:
`02_dnn/08_mnist/tests/test_stage04_reference_payload.py`

QC:
- 5 exact references per dataset if possible.
- all selected refs `train_error == 0`.
- theta length `P=3441`.
- no duplicate theta by L2 distance <= 1e-6.
- if insufficient refs, write blocked report or recommend max_attempt increase / backup architecture.

Stop after Stage 04.


---

## FILE: stages/04_exact_reference_search/CHECKLIST.md

# CHECKLIST — 04_exact_reference_search


- [ ] Reference index exists.
- [ ] Selected theta files exist.
- [ ] Train error recomputation equals stored value.
- [ ] All selected references exact.
- [ ] Reference norm/margin figures exist.
- [ ] Reference caveat included: optimizer-induced, not exact \(P_ref^0\).
- [ ] Pytest passes.
- [ ] `REPORT.md` and `QC_STATUS.json` exist.


---

## FILE: stages/05_pool2_pm_sais_sampling/README.md

# Stage 05 — Pool2 PM-SAIS sampling

## Goal

Run prior-matched angular independent shell sampling for each reference/radius and estimate logZ.

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


---

## FILE: stages/05_pool2_pm_sais_sampling/START_PROMPT.md

GOAL: Execute Stage 05 only: Pool2 PM-SAIS sampling.

Read:
- `02_dnn/08_mnist/01_FORMULAS_PM_SAIS.md`
- `02_dnn/08_mnist/runs/smoke/03_pool_design/POOL_CONTRACT.json`
- `02_dnn/08_mnist/runs/smoke/04_exact_reference_search/reference_index.csv`
- this stage README

Main estimator:
PM-SAIS \(H=\infty\).

Optional diagnostics:
PM-RLI \(H\in\{8,4,2\}\), but main claim remains \(H=\infty\).

Lambda pilot:
Run reduced pilot on:
- all 3 rules
- first split only
- first 3 refs only
- radii `[0.05,0.10,0.20,0.45,0.80]`
- lambda candidates `[1,10,50,100]`

Select a single lambda for all rules using:
- q05 ESS fraction >= 0.02
- split logZ/P <= 0.004
- visible but not saturated phi_energy separation
- no rule-specific lambda

Full smoke radii:
`[0.05,0.10,0.20,0.45,0.80,1.20,1.60,2.00]`

Samples per ref/radius:
- 0.05: 256
- 0.10: 256
- 0.20: 512
- 0.45: 512
- 0.80: 1024
- 1.20: 2048
- 1.60: 2048
- 2.00: 4096

Implement:
- vMF sampling in dimension P.
- stable log_sphere_mgf / logM.
- split samples for split-logZ.
- logsumexp everywhere.
- unit summaries per ref/radius.
- summary-only mode available.

Outputs:
`02_dnn/08_mnist/runs/smoke/05_pool2_pm_sais_sampling/`
- `selected_lambda.json`
- `lambda_selection_report.csv`
- `shell_summary_by_unit.csv`
- `shell_summary_by_rule_radius.csv`
- `qc_by_rule_radius.csv`
- `figures/fig01_lambda_pilot_phi_energy.png`
- `figures/fig02_sampling_qc_ess_heatmap.png`
- `figures/fig03_sampling_qc_split_logz_heatmap.png`
- `figures/fig05_weighted_ce_by_rule_radius.png`
- `REPORT.md`
- `run_config_resolved.json`
- `QC_STATUS.json`

Tests:
Create and run:
`02_dnn/08_mnist/tests/test_stage05_pm_sais_math.py`

QC:
- hard shell distance correct.
- vMF samples unit norm.
- kappa formula correct.
- logZ finite.
- no-claim failed radii.
- bootstrap sd phi <= 0.012 for claimed radii.

Stop after Stage 05.


---

## FILE: stages/05_pool2_pm_sais_sampling/CHECKLIST.md

# CHECKLIST — 05_pool2_pm_sais_sampling


- [ ] Lambda pilot completed.
- [ ] One common lambda selected.
- [ ] PM-SAIS unit summaries exist.
- [ ] QC table exists.
- [ ] Claimed radii pass ESS/split/bootstrap.
- [ ] Failed radii marked no-claim.
- [ ] vMF/hard shell tests pass.
- [ ] `REPORT.md` and `QC_STATUS.json` exist.


---

## FILE: stages/06_results_figures/README.md

# Stage 06 — Results figures and final smoke report

## Goal

Aggregate shell summaries into phi tables, QC tables, and manuscript-style figures.

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


---

## FILE: stages/06_results_figures/START_PROMPT.md

GOAL: Execute Stage 06 only: results aggregation and figures.

Read:
- Stage 01 dataset report.
- Stage 02 complexity report.
- Stage 04 reference report.
- Stage 05 PM-SAIS report.
- `02_dnn/08_mnist/stages/06_results_figures/README.md`

Aggregation:
- Use only QC-passed units.
- Use quenched reference-level average:
  average over split/reference of `[logZ(d)-logZ(d0)]/P`.
- Do not annealed-average Z across references.
- Bootstrap over split/reference units.
- Default `d0=0.05`; if it fails, use smallest common QC-passed radius and report it.

Definitions:
\[
\Delta\phi_{
m energy}(d)=E_{ref}[(\log Z(d)-\log Z(d0))/P].
\]
\[
\Delta\phi_{
m full}(d)=((P-1)/P)\log(d/d0)+\Delta\phi_{
m energy}(d).
\]

Main figure:
`fig04_phi_energy_three_rules_main.png`

Outputs:
`02_dnn/08_mnist/runs/smoke/final_report/`
- `phi_by_rule_radius.csv`
- `phi_bootstrap_by_rule_radius.csv`
- `qc_pass_by_rule_radius.csv`
- `complexity_reference_sampling_joined.csv`
- `final_claim_table.csv`
- `REPORT.md`
- `run_config_resolved.json`
- `QC_STATUS.json`
- figures:
  - `fig01_dataset_montage_28_vs_14.png`
  - `fig02_complexity_nmstv_by_rule.png`
  - `fig03_reference_summary_success_norm_margin.png`
  - `fig04_phi_energy_three_rules_main.png`
  - `fig05_phi_full_three_rules.png`
  - `fig06_area_energy_decomposition.png`
  - `fig07_sampling_qc_pass_heatmap.png`
  - `fig08_sampling_qc_ess_split_bootstrap.png`
  - `fig09_weighted_ce_error_by_radius.png`
  - `fig11_final_storyboard.png`

Figure rules:
- Use matplotlib.
- No seaborn.
- Do not specify custom colors unless existing style requires it.
- Failed/no-claim radii must be omitted or visibly marked.
- Do not plot failed radii as accepted.

Final report must include:
1. Objective.
2. Dataset preparation.
3. Complexity summary.
4. Pool1 reference summary.
5. Pool2 PM-SAIS summary.
6. QC gates.
7. Main phi_energy result.
8. Full phi area-dominance warning.
9. Limitations.
10. Next candidate scale.

Stop after Stage 06.


---

## FILE: stages/06_results_figures/CHECKLIST.md

# CHECKLIST — 06_results_figures


- [ ] `phi_by_rule_radius.csv` exists.
- [ ] `qc_pass_by_rule_radius.csv` exists.
- [ ] `final_claim_table.csv` exists.
- [ ] Main phi_energy figure exists.
- [ ] Full phi figure exists with area caveat.
- [ ] Final report states supported and no-claim radii.
- [ ] Report does not claim production from smoke.
- [ ] `QC_STATUS.json` exists.


---

## FILE: stages/07_candidate_final_promotion/README.md

# Stage 07 — Candidate/final promotion

## Goal

Scale smoke to candidate and final only when previous QC supports promotion.

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


---

## FILE: stages/07_candidate_final_promotion/START_PROMPT.md

GOAL: Execute Stage 07 only if Stage 06 smoke QC passed.

Read:
- `02_dnn/08_mnist/runs/smoke/final_report/REPORT.md`
- `02_dnn/08_mnist/runs/smoke/final_report/final_claim_table.csv`
- this stage README

Candidate:
- identity: `mnist14_3rule_1024_5split_10ref`
- splits: 5
- train size: 1024
- refs per dataset: 10
- use selected lambda from smoke
- use smoke-supported radii
- include 2.80 only as stress if smoke supports 2.00
- outputs under `02_dnn/08_mnist/runs/candidate/`

Final:
- identity: `mnist14_3rule_1024_10split_20ref`
- splits: 10
- train size: 1024
- refs per dataset: 20
- use candidate-selected settings
- outputs under `02_dnn/08_mnist/runs/final/`

Promotion rule:
- Do not promote if smoke/candidate fails QC.
- Do not change architecture/lambda per rule.
- Do not use final language for candidate.

Stop after writing candidate/final promotion report.


---

## FILE: stages/07_candidate_final_promotion/CHECKLIST.md

# CHECKLIST — 07_candidate_final_promotion


- [ ] Smoke QC read.
- [ ] Candidate settings inherit smoke settings.
- [ ] Final settings inherit candidate settings.
- [ ] No rule-specific lambda.
- [ ] Candidate/final outputs separated.
- [ ] Promotion report written.
