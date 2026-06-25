# Updated QC tables

Generated at: 2026-06-18T13:13:58

This update separates sampling completeness from strict claim QC.

- `sampling_complete`: expected rule/ref/radius units exist and have a consistent sample count.
- `diagnostic_phi_available`: sampling is complete and finite enough to use the phi value as a diagnostic curve point.
- `strict_claim_qc_pass`: the old/formal all-reference claim gate, including `max_split_logZ_per_P_diff <= 0.004`.
- `unit_split_fail_count`: number of references at that rule/radius whose single 1024-sample unit exceeded the split gate.

## Run Summary

| run | rows | sampling complete | diagnostic phi available | strict claim pass | unit split fails |
| --- | ---: | ---: | ---: | ---: | ---: |
| 60ref | 100 | 100 | 100 | 6 | 2493 |
| 90ref | 100 | 100 | 100 | 5 | 3734 |

## Global Reference Removal Check

| run | rule | refs total | refs with any split fail | refs remaining after global drop |
| --- | --- | ---: | ---: | ---: |
| 60ref | low_tv_spectral_teacher | 60 | 60 | 0 |
| 60ref | random_label | 60 | 60 | 0 |
| 60ref | real_even_odd | 60 | 60 | 0 |
| 60ref | teacher_nn | 60 | 60 | 0 |
| 90ref | low_tv_spectral_teacher | 90 | 90 | 0 |
| 90ref | random_label | 90 | 90 | 0 |
| 90ref | real_even_odd | 90 | 90 | 0 |
| 90ref | teacher_nn | 90 | 90 | 0 |

## Files

- `derived/qc_table_updated_60ref.csv`
- `derived/qc_table_updated_90ref.csv`
- `derived/qc_table_updated_combined_60_90ref.csv`
- `derived/ref_split_fail_summary_60ref.csv`
- `derived/ref_split_fail_summary_90ref.csv`
- `derived/ref_split_fail_removal_summary_combined_60_90ref.csv`
