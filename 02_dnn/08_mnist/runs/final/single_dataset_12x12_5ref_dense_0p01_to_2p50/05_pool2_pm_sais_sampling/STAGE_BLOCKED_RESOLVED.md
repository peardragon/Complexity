# STAGE_BLOCKED_RESOLVED

Stage: `05_pool2_pm_sais_sampling`

The prior `STAGE_BLOCKED.md` recorded a targeted stability-pilot failure for
`rep8_n2048_cess95_mh2`. The full Stage 05 run was resumed only after a kernel
scan found a passing fallback policy for the remaining unstable case.

Resolution evidence:

- full unit count: `3750`
- Stage 05 final status: `pass`
- all `logZ` and `logZ_inf_full` values finite
- all SMC units completed
- hard-shell max absolute error: `8.881784197001252e-16`
- generated Stage 05 figures: `4`
- fallback policy groups used: `10`
- fallback unit rows: `50`

Residual limitation:

- `716` rule/radius rows are marked `no_claim`; downstream Stage 06 must use
  only QC-passed common radii for claimed phi curves.
