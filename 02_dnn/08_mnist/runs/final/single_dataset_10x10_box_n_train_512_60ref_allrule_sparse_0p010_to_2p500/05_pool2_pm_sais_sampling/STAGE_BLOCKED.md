# Stage 05 Blocked

Status: blocked during sparse-to-2.50 policy iteration on 2026-06-15.

Exact reason: the full Stage05 run was stopped before final aggregation because early low_tv_spectral_teacher unit summaries exceeded the documented split logZ/P gate of 0.004. Continuing that run would have produced a no-claim Stage05 QC failure instead of the requested all-QC-pass phi(d)_energy line.

Observed failing units:

- low_tv_spectral_teacher ref_001 d_raw=1.7500 with sparse_rep16_n2048_cess95_mh2: split_logZ_per_P_diff=0.01085798177361284.
- low_tv_spectral_teacher ref_001 d_raw=2.0000 with sparse_rep8_n2048_cess95_mh2: split_logZ_per_P_diff=0.004129318355180518.
- low_tv_spectral_teacher ref_029 d_raw=1.2500 with baseline sampling: split_logZ_per_P_diff=0.005777577415930825.
- low_tv_spectral_teacher ref_013 d_raw=0.4500 with baseline sampling: split_logZ_per_P_diff=0.005303409591476637.
- low_tv_spectral_teacher ref_001 d_raw=1.0000 with sparse_rep16_n2048_cess95_mh2: split_logZ_per_P_diff=0.005119200047769858.
- low_tv_spectral_teacher ref_002 d_raw=1.0000 with sparse_rep16_n4096_cess95_mh2: split_logZ_per_P_diff=0.00453803722026287.
- low_tv_spectral_teacher ref_000 d_raw=1.7500 with sparse_rep32_n4096_cess95_mh2: split_logZ_per_P_diff=0.005157646432945215.
- low_tv_spectral_teacher ref_002 d_raw=2.2500 with sparse_rep16_n4096_cess95_mh2: split_logZ_per_P_diff=0.007599867845450754.
- low_tv_spectral_teacher ref_002 d_raw=2.5000 with sparse_rep16_n4096_cess95_mh2: split_logZ_per_P_diff=0.00602180993556809.
- low_tv_spectral_teacher ref_000 d_raw=1.2500 with sparse_rep64_n4096_cess95_mh2: split_logZ_per_P_diff=0.005264728392728507.
- low_tv_spectral_teacher ref_001 d_raw=2.0000 with sparse_rep64_n4096_cess95_mh2: split_logZ_per_P_diff=0.004982366690901986.
- low_tv_spectral_teacher ref_000 d_raw=1.7500 with sparse_rep64_n4096_cess95_mh2: split_logZ_per_P_diff=0.004745377272441687.

Next safe action: do not promote Stage05 or Stage06 outputs from this interrupted run. Treat the current sparse-to-2.50 attempt as blocked beyond the already completed/pass dense overlap radii. The next safe methodological step is to decide whether the single-dataset, many-reference QC contract should remain an all-reference max split gate or be changed explicitly to a reference-level or robust aggregate QC contract; simply increasing the current PM-SAIS sample policy again is not justified because sparse_rep64_n4096_cess95_mh2 still exceeded the split gate. Existing retained production outputs must remain unchanged.
