# Stage Blocked: `dense_qc_stable_ref30` at `d_raw=0.85`

The targeted pilot was stopped after a QC failure.

Failing unit:
`split_000/low_tv_spectral_teacher/ref_027/r_0p8500`

Observed metrics:

- `split_logZ_per_P_diff = 0.005223599532885898`
- split gate: `0.004`
- `ess_fraction = 0.7729716920403772`
- `replicates = 16`
- `n_samples_total = 32768`
- fallback policy: `sparse_rep16_n2048_cess95_mh2`
- `smc_completed = true`
- `smc_min_cess_fraction = 0.9500000000000729`
- elapsed seconds: `394.2435531616211`

The source row for the same reference/radius already failed split QC (`0.004421218572654 > 0.004`), and the forced targeted rerun still failed. This blocks promotion of a single averaged `dense_qc_stable_ref30` curve at `d_raw=0.85`.

Next safe action: keep the complete/pass curve only through `d_raw=0.65`, then perform family-boundary analysis before any larger sparse-domain production run.
