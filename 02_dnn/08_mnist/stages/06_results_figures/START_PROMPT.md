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
\Delta\phi_{m energy}(d)=E_{ref}[(\log Z(d)-\log Z(d0))/P].
\]
\[
\Delta\phi_{m full}(d)=((P-1)/P)\log(d/d0)+\Delta\phi_{m energy}(d).
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
