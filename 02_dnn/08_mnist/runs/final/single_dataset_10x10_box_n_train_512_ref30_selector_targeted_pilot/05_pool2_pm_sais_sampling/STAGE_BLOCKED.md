# Stage Blocked: Targeted Stage05 Selector QC

Run root:
`02_dnn/08_mnist/runs/final/single_dataset_10x10_box_n_train_512_ref30_selector_targeted_pilot`

Stage:
`05_pool2_pm_sais_sampling`

Selector:
`dense_qc_stable_ref30`

Rule:
`low_tv_spectral_teacher`

Blocked radius:
`d_raw = 0.85`

## Exact Reason

The targeted Stage05 pilot for `d_raw=0.85` hit a QC failure and was stopped.

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

The failure is split-logZ stability, not ESS. This is a repeated hard signal for the single-family `dense_qc_stable_ref30` claim at `d_raw=0.85`: the source row for ref027 at the same radius also failed split QC (`0.004421218572654 > 0.004`), and the forced targeted rerun still failed (`0.005223599532885898 > 0.004`).

The active targeted pilot process was terminated after the failure was observed. No retained production outputs were modified.

## Completed Before Stop

The same `d_raw=0.85` targeted pilot completed:

- ref022: pass, `split_logZ_per_P_diff = 0.0002678328598322561`, `ess_fraction = 0.8098776353491718`
- ref027: fail, `split_logZ_per_P_diff = 0.005223599532885898`, `ess_fraction = 0.7729716920403772`

The run had just started ref030 when it was stopped.

## Next Safe Action

Do not promote `d_raw=0.85` or launch larger sparse-domain production for a single averaged `dense_qc_stable_ref30` curve.

The next safe analysis step is to treat `d_raw=0.85` as a candidate family-boundary radius:

1. Freeze the currently claimable single-family curve at radii that passed complete selector QC: `0.01, 0.02, 0.03, 0.05, 0.08, 0.45, 0.65`.
2. Perform reference-family decomposition using per-reference `phi(d)_energy`, split-logZ hardness, parameter/function distance, and stability diagnostics.
3. Separate observed boundary-failing references from the current single-family claim unless a predeclared stronger reproducibility check is explicitly run and passes.
4. If continuing diagnostically, run only a narrow, predeclared hard-reference audit at `d_raw=0.85`; do not use it as production promotion evidence until the family definition is updated and selector-level QC is recomputed.

## Boundary Analysis Artifacts

Follow-up analysis was written to:

`02_dnn/08_mnist/runs/final/single_dataset_10x10_box_n_train_512_ref30_selector_targeted_pilot/07_family_boundary_analysis/`

Key artifacts:

- `REPORT.md`
- `claimable_phi_curve.csv`
- `boundary_reference_diagnostics.csv`
- `large_domain_decision.json`
- `candidate_selector_qc_status.csv`
- `candidate_recovery_tasks.csv`
- `figures/fig01_claimable_phi_energy_curve.png`
- `figures/fig02_reference_phi_energy_spaghetti_boundary.png`
- `figures/fig03_split_logz_heatmap_boundary.png`
- `figures/fig04_reference_family_pca.png`
- `figures/fig05_candidate_ref29_qc_coverage.png`

## Diagnostic Ref29 Candidate

A diagnostic candidate selector was evaluated without additional sampling:

`ref29_minus_boundary_fail_ref027`

This candidate removes ref027 from the original `dense_qc_stable_ref30` selector. It is not promoted.

At `d_raw=0.85`, the initial evidence before the missing-fill diagnostic was:

- selected refs: `29`
- observed refs: `12`
- missing refs: `17`
- observed fail refs: none
- status: `missing_only`
- missing refs: `30,31,32,33,38,40,41,42,43,44,46,49,52,55,57,58,59`

This is only a possible recovery path. If the updated family law is explicitly accepted before execution, the next safe diagnostic task is limited to the missing 17 references at `d_raw=0.85`. Larger radii must remain blocked until `d_raw=0.85` is complete and selector-level QC passes.

## Diagnostic Ref29 Missing-Fill Failure

The diagnostic missing-fill task for `ref29_minus_boundary_fail_ref027` was started at `d_raw=0.85` for:

`30,31,32,33,38,40,41,42,43,44,46,49,52,55,57,58,59`

It was stopped after ref033 failed split-logZ QC.

Completed diagnostic units before stop:

- ref030: pass, `split_logZ_per_P_diff = 0.0014365467332433325`, `ess_fraction = 0.8004711822550727`
- ref031: pass, `split_logZ_per_P_diff = 0.002404749103812401`, `ess_fraction = 0.805868693942769`
- ref032: pass, `split_logZ_per_P_diff = 0.0037773101030130914`, `ess_fraction = 0.8646707189196408`
- ref033: fail, `split_logZ_per_P_diff = 0.005032340321810607`, `ess_fraction = 0.8142279414568929`

The ref033 failure is also split-logZ stability, not ESS:

- split gate: `0.004`
- `replicates = 16`
- `n_samples_total = 32768`
- fallback policy: `sparse_rep16_n2048_cess95_mh2`
- `smc_completed = true`
- `smc_min_cess_fraction = 0.950000000000056`

Therefore `ref29_minus_boundary_fail_ref027` is also not promotable as a large-domain recovery selector. Any further recovery must define a new predeclared family law, for example excluding both ref027 and ref033, before running more diagnostics.

## Diagnostic Ref28 Missing-Fill Failure

An exploratory candidate family law was recorded in:

`02_dnn/08_mnist/runs/final/single_dataset_10x10_box_n_train_512_ref30_selector_targeted_pilot/07_family_boundary_analysis/EXPLORATORY_CANDIDATE_REF28.md`

Candidate:

`minus_all_observed_boundary_fail_refs`

At the time of definition this removed ref027 and ref033, leaving 28 selected references. The diagnostic missing-fill task for `d_raw=0.85` was started for:

`38,40,41,42,43,44,46,49,52,55,57,58,59`

It was stopped after ref049 failed split-logZ QC.

Completed diagnostic units before stop:

- ref038: pass, `split_logZ_per_P_diff = 0.003122696507484516`, `ess_fraction = 0.8312676284609908`
- ref040: pass, `split_logZ_per_P_diff = 0.0007192402452124484`, `ess_fraction = 0.7555577922205509`
- ref041: pass, `split_logZ_per_P_diff = 0.0004802434556782324`, `ess_fraction = 0.837171280519386`
- ref042: pass, `split_logZ_per_P_diff = 0.0017738977108755528`, `ess_fraction = 0.850321786583022`
- ref043: pass, `split_logZ_per_P_diff = 0.0013028790916945199`, `ess_fraction = 0.8126813386221828`
- ref044: pass, `split_logZ_per_P_diff = 0.00016093015741427448`, `ess_fraction = 0.8382417795463375`
- ref046: pass, `split_logZ_per_P_diff = 0.0017338027062689968`, `ess_fraction = 0.8254087052369908`
- ref049: fail, `split_logZ_per_P_diff = 0.005459735346338214`, `ess_fraction = 0.7592843037199755`

The ref049 failure is also split-logZ stability, not ESS:

- split gate: `0.004`
- `replicates = 16`
- `n_samples_total = 32768`
- fallback policy: `sparse_rep16_n2048_cess95_mh2`
- `smc_completed = true`
- `smc_min_cess_fraction = 0.9500000000001387`

Therefore the exploratory ref28 candidate is also not promotable. The currently observed boundary-failing references at `d_raw=0.85` are ref027, ref033, and ref049.
