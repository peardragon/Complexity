# Project Major Instructions

## Active Layout Contract

This repository is organized around the current promoted theory and 3NN
results. The active tree is intentionally compact:

- Root-level retained directories are `.venv`, `01_theory`, `02_dnn`,
  `99_backup`, and `99_codex_bin`.
- Each retained stage uses `config/`, `src/`, `raw_outputs/`, `figures/`, and
  optional `QC/`.
- `summary/`, `results_summary/`, `results_raw/`, and dimensioned `d2/`
  directories are legacy contracts. Do not create new active final outputs
  there.
- Old run hashes may appear only inside explicit provenance files.

## Smoke Runs Contract

Every retained stage has a top-level `smoke_runs/` directory.

- `smoke_runs/` is for smoke checks, short exploratory runs, test beds, and
  temporary additional experiments.
- Smoke results are not promoted final results until explicitly copied into the
  stage's active `raw_outputs/`, `figures/`, or `QC/` path with provenance.
- Use user-friendly run names under `smoke_runs/{run_info}/`, not hashes.
- A smoke run repeats the stage shape:
  - analytic/comparison stages: `config/`, `src/`, `raw_outputs/` when needed,
    and `figures/`
  - sampling stages: `config/`, `src/`, `raw_outputs/`, `figures/`, and `QC/`
  - proxy-only stages: `config/`, `src/`, and `figures/`

## Promoted Finals

- Theory final:
  - sampling source/result identity:
    `two_pool_perceptron_alpha0p1_dense_qc_full_split_pass_merged`
  - comparison figure:
    `01_theory/03_theory_comparison/figures/fig00_dense_qc_N_convergence_alpha0p1.png`
  - retained sampling pools:
    `01_theory/02_theory_sampling/raw_outputs/reference_pool/` and
    `01_theory/02_theory_sampling/raw_outputs/shell_pool/`
- 3NN final:
  - final proxy source identity:
    `9_beta_cell_10_dataset_10_reference_1024_sample_per_ref_radius_SMC_pool1_l2top10_lambda1_gamma1`
  - active public path prefix:
    `02_dnn/05_proxy_local_entropy/figures/9_beta_cell_10_dataset_10_reference/`
  - retained ranges:
    `d_0.01_to_0.10`, `d_0.10_to_3.00`, `d_0.01_to_2.50_dense`,
    and `d_0.01_to_2.50_sparse`.

## Cleanup Boundary

- Before removing non-retained material, move it under
  `99_backup/cleanup_<timestamp>/`.
- Codex, one-off helper, old runner, old tool, payload, research, and temporary
  root artifacts belong under `99_codex_bin/cleanup_<timestamp>/`.
- Do not permanently delete ignored raw/result/cache payloads unless the user
  explicitly asks for permanent deletion.
- Keep rollback metadata in
  `99_backup/cleanup_<timestamp>/cleanup_manifest.json`.

## Execution Timing Rule

Before running an expensive default/full experiment, run the matching smoke
preset once and estimate wall time from the measured trial count or attempted
unit count.

- If the estimated runtime is not day-scale, do not impose a command timeout
  that can terminate the run early.
- For long but sub-day runs, use a no-timeout/background execution pattern and
  monitor progress until completion.
- Keep the measured smoke time, estimate, and actual elapsed time in the
  relevant recovery, cleanup, or run report when the run changes retained
  results.
