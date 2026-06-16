# MNIST10 Family Boundary Analysis

Selector: `dense_qc_stable_ref30`

Rule: `low_tv_spectral_teacher`

## Decision

The current evidence supports a single-family averaged `phi(d)_energy` curve only through:

`0.01, 0.02, 0.03, 0.05, 0.08, 0.45, 0.65`

`d_raw=0.85` is blocked for the current single-family selector. Boundary-failing references observed so far: `ref027, ref033, ref049`.

## Blocked Radius Summary

- blocked radius: `0.85`
- selector QC pass: `False`
- claim status: `missing_units_and_qc_fail`
- observed refs at blocked radius: `25`
- max split logZ/P diff at blocked radius: `0.0054597353463382`

## Boundary Reference

| ref | split at 0.85 | ESS at 0.85 | status |
| --- | ---: | ---: | --- |
| 27 | 0.005223599533 | 0.772972 | boundary_fail_0p85 |
| 33 | 0.005032340322 | 0.814228 | boundary_fail_0p85 |
| 49 | 0.005459735346 | 0.759284 | boundary_fail_0p85 |

## Interpretation

The `dense_qc_stable_ref30` selector behaves like a coherent single family up to `d_raw=0.65`; both `d_raw=0.45` and `d_raw=0.65` are complete 30-reference QC-pass rows after targeted overlay. At `d_raw=0.85`, ref027 fails split-logZ stability in both the source row and the forced targeted rerun. A diagnostic ref29-minus-ref027 recovery attempt then found another boundary failure at ref033. These failures are not ESS failures.

This means sparse large-domain production should not be promoted as one averaged ref30 curve from the current selector. The next safe step is family decomposition: either define a predeclared updated family law that excludes the observed boundary-failing references and rerun selector QC, or report separate family curves once enough large-radius rows exist for each family.

## Candidate Ref29 Recovery State

Candidate: `ref29_minus_boundary_fail_ref027`

This candidate is diagnostic only. It removes the repeated boundary-failing ref027 from the original 30-reference selector.

At `d_raw=0.85`:

- selected refs: `29`
- observed refs: `24`
- missing refs: `5`
- observed fail refs: `33;49`
- status: `missing_and_observed_fail`

No missing-fill task is safe for this candidate as a promotion path, because it already has an observed failure at `d_raw=0.85`: `33;49`. The next safe action is to define a new predeclared family law that excludes the observed boundary-failing references, or to run a separate hard-reference audit explicitly labeled as diagnostic.

## Artifacts

- `claimable_phi_curve.csv`
- `boundary_reference_diagnostics.csv`
- `selector_qc_status.csv`
- `family_cluster_assignments.csv`
- `candidate_selector_qc_status.csv`
- `candidate_recovery_tasks.csv`
- `large_domain_decision.json`
- `figures/fig01_claimable_phi_energy_curve.png`
- `figures/fig02_reference_phi_energy_spaghetti_boundary.png`
- `figures/fig03_split_logz_heatmap_boundary.png`
- `figures/fig04_reference_family_pca.png`
- `figures/fig05_candidate_ref29_qc_coverage.png`
