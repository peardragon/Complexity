# Replicate Stability Probe

Generated at: 2026-06-18T12:00:15

## Design

- Runs fresh independent SMC replicates for selected `rule/ref/radius` units.
- This is the appropriate follow-up when arbitrary random re-split logZ cannot be reconstructed from saved normalized sample weights.
- Replicates per probe unit: `2`.
- Samples per replicate: `1024`.
- Probe radii: `0.3`.
- Refs per rule/radius cell: `2` selected by source split rank.

## Current Run

- Probe units selected: `2`.
- Replicate rows completed: `4`.
- Total elapsed SMC seconds: `65.3`.

## Unit Summary

| rule | radius | ref | label | phi sd | split q95 | split fail rate |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| low_tv_spectral_teacher | 0.3 | 1028 | high_split | 0.00202333 | 0.00286915 | 0.000 |
| low_tv_spectral_teacher | 0.3 | 16 | low_split | 0.000939303 | 0.0033578 | 0.000 |

## Outputs

- `probe_tasks.csv`
- `replicate_unit_results.csv`
- `replicate_unit_summary.csv`
- `replicate_rule_radius_summary.csv`
