# Replicate Stability Probe

Generated at: 2026-06-18T12:22:52

## Design

- Runs fresh independent SMC replicates for selected `rule/ref/radius` units.
- This is the appropriate follow-up when arbitrary random re-split logZ cannot be reconstructed from saved normalized sample weights.
- Replicates per probe unit: `2`.
- Samples per replicate: `1024`.
- Probe radii: `0.3,1.0,2.5`.
- Refs per rule/radius cell: `2` selected by source split rank.

## Current Run

- Probe units selected: `24`.
- Replicate rows completed: `48`.
- Total elapsed SMC seconds: `1259.3`.

## Unit Summary

| rule | radius | ref | label | phi sd | split q95 | split fail rate |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| low_tv_spectral_teacher | 0.3 | 1028 | high_split | 0.00202333 | 0.00286915 | 0.000 |
| low_tv_spectral_teacher | 0.3 | 16 | low_split | 0.000939303 | 0.0033578 | 0.000 |
| low_tv_spectral_teacher | 1.0 | 52 | high_split | 0.000201625 | 0.0106552 | 0.500 |
| low_tv_spectral_teacher | 1.0 | 57 | low_split | 0.00110054 | 0.00468529 | 1.000 |
| low_tv_spectral_teacher | 2.5 | 46 | high_split | 0.00363842 | 0.0139102 | 0.500 |
| low_tv_spectral_teacher | 2.5 | 1014 | low_split | 0.00456471 | 0.00592866 | 1.000 |
| random_label | 0.3 | 36 | high_split | 0.00140443 | 0.00865745 | 0.500 |
| random_label | 0.3 | 1018 | low_split | 0.00619436 | 0.0060023 | 1.000 |
| random_label | 1.0 | 1011 | high_split | 0.000331379 | 0.0100315 | 1.000 |
| random_label | 1.0 | 39 | low_split | 0.00113263 | 0.00489776 | 0.500 |
| random_label | 2.5 | 18 | high_split | 0.000433508 | 0.00319855 | 0.000 |
| random_label | 2.5 | 1001 | low_split | 0.00680085 | 0.0100126 | 0.500 |
| real_even_odd | 0.3 | 1008 | high_split | 0.000826734 | 0.0023451 | 0.000 |
| real_even_odd | 0.3 | 1020 | low_split | 0.000793679 | 0.000991452 | 0.000 |
| real_even_odd | 1.0 | 46 | high_split | 0.00200549 | 0.00501654 | 0.500 |
| real_even_odd | 1.0 | 21 | low_split | 0.00340125 | 0.00188907 | 0.000 |
| real_even_odd | 2.5 | 1008 | high_split | 0.00286825 | 0.00251399 | 0.000 |
| real_even_odd | 2.5 | 34 | low_split | 0.00351547 | 0.0114455 | 0.500 |
| teacher_nn | 0.3 | 45 | high_split | 0.00055929 | 0.00132101 | 0.000 |
| teacher_nn | 0.3 | 23 | low_split | 0.000629271 | 0.00209764 | 0.000 |
| teacher_nn | 1.0 | 1025 | high_split | 0.00358988 | 0.00683313 | 0.500 |
| teacher_nn | 1.0 | 1003 | low_split | 0.002418 | 0.00152481 | 0.000 |
| teacher_nn | 2.5 | 14 | high_split | 0.000327353 | 0.00255923 | 0.000 |
| teacher_nn | 2.5 | 1001 | low_split | 0.00479156 | 0.00208514 | 0.000 |

## Outputs

- `probe_tasks.csv`
- `replicate_unit_results.csv`
- `replicate_unit_summary.csv`
- `replicate_rule_radius_summary.csv`
