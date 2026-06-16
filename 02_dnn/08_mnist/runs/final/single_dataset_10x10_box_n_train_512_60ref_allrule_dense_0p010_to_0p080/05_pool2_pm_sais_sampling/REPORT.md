# Stage 05 PM-SAIS Sampling

Completed 3360 shell units with 853 fallback-policy units.

Sample counts present: [4096]

Fallback total-sample counts: {16384: 721, 32768: 57, 65536: 75}

Sampling note: All-rule dense extension. Reuses identical 10x10 BOX datasets, exact references, strong microline r0 units, broad endpoint units, and the completed low-TV dense run where available. Known bad broad outliers are intentionally not copied.

All selected rule/radius rows passed the Stage 05 QC gates.

## Runtime And Reuse

Stage 05 used 3360 unit summaries: 2436 copied/reused unit JSON payloads and 924 newly computed unit JSON payloads. Newly computed unit elapsed time summed to 4326.89 s across unit payloads.

Newly computed units by rule:

| rule | copied units | newly computed units | new-unit elapsed s |
| --- | ---: | ---: | ---: |
| low_tv_spectral_teacher | 840 | 0 | 0.00 |
| random_label | 518 | 322 | 2679.14 |
| real_even_odd | 538 | 302 | 559.93 |
| teacher_nn | 540 | 300 | 1087.82 |

Five known bad broad-run outlier units were intentionally not copied and were recomputed with `rep16_n4096_cess95_mh2_outlier_recompute`: `random_label/ref_004/r=0.0200`, `random_label/ref_007/r=0.0300`, `random_label/ref_018/r=0.0300`, `real_even_odd/ref_000/r=0.0500`, and `real_even_odd/ref_027/r=0.0500`.
