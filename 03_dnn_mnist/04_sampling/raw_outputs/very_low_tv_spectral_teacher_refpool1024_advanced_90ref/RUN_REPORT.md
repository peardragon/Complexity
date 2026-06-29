# Advanced 90ref Sampling Run Report

Run root:
`04_sampling/raw_outputs/very_low_tv_spectral_teacher_refpool1024_advanced_90ref`

## Scope

- Radius grid: advanced, `0.10, 0.15, ..., 2.50` (`49` radii).
- Active rules: `very_low_tv_spectral_teacher`, `real_even_odd`, `teacher_nn`, `random_label`.
- Deprecated/excluded rule: `low_tv_spectral_teacher`.
- Target references: `90` per rule.
- Expected units: `4 rules * 90 refs * 49 radii = 17640`.
- Final status: complete, `17640 / 17640` units.

## Resource Policy

- User limit: CPU 35%, GPU 25%.
- Run policy: CPU device, non-overlap taskset shard placement.
- Logical CPUs: `32`.
- Cap used by manager: `11` logical CPUs, `10` shards, `1` thread per shard, one two-core shard.
- Observed run stayed below the CPU cap; GPU use was effectively `0%`.

## Smoke Timing And Estimate

Smoke root:
`04_sampling/smoke_runs/very_low_advanced_sampler_smoke`

Measured very-low single-reference smoke timings:

| radius | elapsed seconds |
| ---: | ---: |
| 0.10 | 5.635 |
| 0.15 | 9.295 |
| 1.25 | 30.334 |
| 2.45 | 36.684 |

Before launching the capped final manager, the estimated remaining wall time was
approximately `2-3 hours` under the 35% CPU cap. The run reused validated
cached/prepopulated units from prior production outputs and generated the
remaining very-low advanced units plus the final aggregate.

Final manager timing:

- Started: `2026-06-25T22:06:08+09:00`.
- Finished: `2026-06-26T00:23:53+09:00`.
- Elapsed: `2:17:45`.
- Exit status: `0`.

## Internal Derivative Outputs

- Unit table with internal phi derivatives:
  `04_sampling/raw_outputs/very_low_tv_spectral_teacher_refpool1024_advanced_90ref/05_pool2_pm_sais_sampling/shell_summary_by_unit_with_phi_derivatives.csv`
- Rule-radius derivative table:
  `04_sampling/raw_outputs/very_low_tv_spectral_teacher_refpool1024_advanced_90ref/06_results_figures/dphi_dd_by_rule_radius.csv`
- Rule-radius phi table:
  `04_sampling/raw_outputs/very_low_tv_spectral_teacher_refpool1024_advanced_90ref/06_results_figures/phi_by_rule_radius.csv`

Verified rows:

- Unit derivative table: `17640`.
- Phi rule-radius table: `196`.
- dphi rule-radius table: `196`.

## Promoted Figure Locations

Dataset label/representation outputs:

- Raw tables/status:
  `01_dataset_gen/raw_outputs/active_rule_dataset_representations_very_low_refpool1024_advanced_90ref`
- Figures:
  `01_dataset_gen/figures/active_rule_dataset_representations_very_low_refpool1024_advanced_90ref`

Phi/proxy/local-entropy outputs:

- Raw tables/status:
  `05_proxy_local_entropy/raw_outputs/very_low_tv_spectral_teacher_refpool1024_advanced_90ref`
- Figures:
  `05_proxy_local_entropy/figures/very_low_tv_spectral_teacher_refpool1024_advanced_90ref`

The active `04_sampling/.../06_results_figures` directory is retained only for
aggregate CSV provenance used by the sampling status, not as a promoted figure
location.

## Legacy-Style Figure Set

The old backup bundle under
`99_backup/cleanup_20260626_002622/04_sampling/raw_outputs/very_low_tv_spectral_teacher_refpool1024_90ref/06_results_figures`
contained active-rule NMSTV spaghetti, dataset-label examples, and t-SNE label
embedding figures. The same figure style was regenerated for this advanced
0.05-grid run while keeping the promoted stage split:

- Phi/NMSTV legacy-style figures:
  `05_proxy_local_entropy/figures/very_low_tv_spectral_teacher_refpool1024_advanced_90ref/legacy_style`
- Phi/NMSTV legacy-style tables/status:
  `05_proxy_local_entropy/raw_outputs/very_low_tv_spectral_teacher_refpool1024_advanced_90ref/legacy_style`
- Dataset legacy-style figures:
  `01_dataset_gen/figures/active_rule_dataset_representations_very_low_refpool1024_advanced_90ref/legacy_style`
- Dataset legacy-style tables/status:
  `01_dataset_gen/raw_outputs/active_rule_dataset_representations_very_low_refpool1024_advanced_90ref/legacy_style`

Builder:
`04_sampling/src/build_advanced_legacy_style_figures.py`

## Cleanup

Misplaced or deprecated result bundles were moved to:
`99_backup/cleanup_20260626_002622`

Rollback metadata:
`99_backup/cleanup_20260626_002622/cleanup_manifest.json`

No files were permanently deleted.
