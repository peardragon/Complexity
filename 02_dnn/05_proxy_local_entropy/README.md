# DNN Proxy Local Entropy

This stage converts shell-sampling summaries into compact proxy local-entropy
tables and interactive figures.

## Method

- local-entropy proxy: shell area term plus weighted shell log-partition terms
- full L2 view: uses `logZ_inf_full`, `logZ_inf_stripped`, and
  `reference_prior_log_weight` when present
- fallback regularization: `compute_phi.DEFAULT_LAMBDA_REG=220` only when full
  fields are absent
- table code: `src/make_proxy_tables.py`, `src/compute_phi.py`
- dashboard code: `src/build_interactive_dashboard.py`
- legacy/static plotting code retained for provenance:
  `src/plot_summary.py`, `src/plot_summary_derivative.py`

## Active Outputs

Production run:

- `raw_outputs/18_beta_cell_30_dataset_30_reference/d_0.01_to_2.50_dense/`
- `figures/18_beta_cell_30_dataset_30_reference/d_0.01_to_2.50_dense/local_entropy_dashboard.html`

Completed extension:

- `raw_outputs/18_beta_cell_60_dataset_30_reference/d_0.01_to_2.50_dense/`
- `figures/18_beta_cell_60_dataset_30_reference/d_0.01_to_2.50_dense/local_entropy_dashboard.html`

Combined analysis dashboard:

- `figures/local_entropy_dashboard.html`

The combined dashboard replaces the old mixed static figure set. It provides
the same controls for both active runs: metric family, metric, beta subset,
radius window, linear/log radius scale, signed-log value transform, and
line/heatmap view.

## Table Families

- `absolute_phi_by_beta_radius.csv`: absolute `phi(d)` and energy/area split
- `delta_phi_by_beta_radius.csv`: `phi(d)-phi(r0)` and energy/area split
- `dphi_dr_by_beta_radius.csv`: radial derivatives
- `hq_by_beta_radius.csv`: H-threshold phase-map inputs
- `accuracy_q_by_beta_radius.csv`: optional accuracy phase-map input, generated
  only when every sample payload is reread

## Figure Contract

- The active 05-stage analysis figure is `figures/local_entropy_dashboard.html`.
- Per-run dashboards are retained only to inspect one run without manually
  disabling the other run.
- Static PNG/CSV/one-off report outputs from older figure passes were archived
  under `99_backup/cleanup_20260610_160312/`.
- Rebuild the dashboards with:

```powershell
python 02_dnn/05_proxy_local_entropy/src/build_interactive_dashboard.py --repo-root .
```
