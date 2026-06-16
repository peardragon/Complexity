# STAGE_BLOCKED

Stage: `07_reference_family_analysis` sparse large-domain continuation decision.

## Exact Failing Condition

Sparse large-domain production was not launched because no predeclared 30-reference low_tv selector has complete and QC-passing evidence over the large-domain radii.

## Observed

- `large_domain_supported_for_predeclared_lowtv_ref30`: `False`
- `optimizer_first30_ref30` first low_tv no-claim radius: `0.45`
- `optimizer_first30_ref30` first low_tv no-claim status: `no_claim_qc_fail`
- `optimizer_first30_ref30` first low_tv no-claim max split logZ/P: `0.005303409591476637`
- split gate: `0.004`
- `l2_min_norm_ref30` first large-radius observed refs: `13 / 30`
- `l2_min_norm_ref30` first large-radius missing refs: `17`
- `l2_min_norm_ref30` first large-radius status: `missing_units`

## Expected

- A predeclared 30-reference low_tv selector must have complete selected-reference units at every large radius.
- Every selected selector/radius row must satisfy split logZ/P <= `0.004`, q05 ESS >= `0.04`, and bootstrap sd phi <= `0.012`.

## Next Safe Action

Do not launch sparse large-domain production from the current evidence. First define a predeclared reference law such as l2_min_norm_ref30, run a targeted Stage05 pilot on its missing/hard large-radii units, and only promote if complete selector-level QC passes.

Do not promote sparse large-domain phi(d)_energy figures from incomplete/no-claim rows.
