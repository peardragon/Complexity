# Exploratory Candidate: `minus_all_observed_boundary_fail_refs`

This is an exploratory family-boundary diagnostic, not a promoted production selector.

## Definition

Base selector:
`dense_qc_stable_ref30`

Rule:
`low_tv_spectral_teacher`

Blocked radius:
`d_raw = 0.85`

Observed boundary-failing references removed:

- ref027
- ref033

Candidate selector:
`minus_all_observed_boundary_fail_refs`

Selected reference count:
`28`

## Current Evidence Before Missing-Fill

At `d_raw=0.85`:

- observed refs: `15`
- missing refs: `13`
- observed fail refs: none
- max observed split logZ/P excluding ref027/ref033: `0.003777310103013`
- q05 ESS: `0.6572097611501009`
- status: `missing_only`

Missing refs for the next diagnostic-only fill:

`38,40,41,42,43,44,46,49,52,55,57,58,59`

## Execution Rule

This candidate may only be used as a diagnostic recovery test at `d_raw=0.85`.

Stop immediately on any QC failure and update `STAGE_BLOCKED.md`. Do not use this candidate to promote sparse large-domain production unless `d_raw=0.85` is complete and selector-level QC passes, and a subsequent predeclared large-radius plan is written before execution.
