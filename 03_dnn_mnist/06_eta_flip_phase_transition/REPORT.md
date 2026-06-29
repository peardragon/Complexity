# Eta Flip Phase Transition Work Report

## Scope Completed

This stage explores MNIST even/odd label-flip noise `eta` as a complexity knob.
The current outputs are intentionally split into evidence levels:

1. `eta_sweep_pilot_cpu35_gpu0`
   - CPU-only graph-complexity proxy.
   - Uses nested flip masks over the fixed even/odd dataset.
   - Computes kNN normalized MSTV and first-derivative proxy over eta.
   - No DNN sampling is involved.

2. `eta_anchor_phi_short_d1_cpu35_gpu0`
   - CPU-only phi(d) anchor smoke near d=1.
   - Reuses one existing `real_even_odd` reference as a fixed anchor.
   - Evaluates eta-flipped datasets around that anchor.
   - This is not an eta-specific reference-search result.

3. `eta_reference_search_4eta_3ref_cpu35_gpu0`
   - Eta-specific exact-reference search smoke.
   - Eta values: `{0.0, 0.2, 0.35, 0.5}`.
   - Three exact references per eta were selected.

4. `eta_reference_phi_4eta_3ref_d1_n128_cpu35_gpu0`
   - Eta-specific reference phi smoke.
   - Eta values: `{0.0, 0.2, 0.35, 0.5}`.
   - Three references per eta, n=128 samples per unit.
   - Useful for reference averaging, but some split-logZ diagnostics are noisy.

5. `eta_reference_phi_4eta_1ref_d1_n1024_cpu35_gpu0`
   - Eta-specific reference phi confirmation.
   - Eta values: `{0.0, 0.2, 0.35, 0.5}`.
   - One reference per eta, n=1024 samples per unit.
   - Useful for sampler precision, but not for reference-family averaging.

6. `eta_reference_phi_dense_eta_ref1_d1_n1024_cpu35_gpu0`
   - Dense eta zoom with eta values `{0.2, 0.25, 0.3, 0.35, 0.4, 0.5}`.
   - One reference per eta, n=1024 samples per unit.
   - Radii `{0.8, 0.9, 1.0, 1.1}`; raw phi and first derivatives only.

7. `eta_reference_search_promoted_4eta_10ref_cpu35_gpu0`
   - Promoted eta-specific exact-reference search.
   - Eta values `{0.25, 0.30, 0.35, 0.40}`.
   - Ten exact references per eta; 40/40 references selected.

8. `eta_reference_phi_promoted_4eta_10ref_d1_n1024_cpu35_gpu0`
   - Promoted eta-specific phi run, extended to the small-d range.
   - Eta values `{0.25, 0.30, 0.35, 0.40}`.
   - Ten references per eta, n=1024 samples per unit.
   - Active summarized radii `{0.1, 0.2, ..., 1.0}`.
   - Older `r=1.1` unit files remain in `unit_summaries/`, but the active
     summary and curvature figures filter to the requested `0.1..1.0` grid.
   - Ref-level first derivatives are computed before averaging.

## Main Findings So Far

- k=3 graph NMSTV rises from about 0.30 at eta=0 to about 1.00 at eta=0.5.
- k=3 crosses NMSTV>=0.90 around eta=0.35.
- The first-derivative proxy peaks near eta=0, not at an interior eta.
- For independent label flips, the graph-TV expectation is closed form:
  `q_eta = q0 + 2 eta (1 - eta) (1 - 2 q0)`.
- Therefore a true interior phase-transition signal, if present, should come
  from DNN sampling/reference geometry rather than from graph-TV algebra alone.
- Eta-specific reference search was easy in the current smoke settings:
  all requested exact references were found within the first four attempts per eta.
- The eta-specific phi(d=1) raw energy drops sharply between eta=0.20 and
  eta=0.35, then mostly saturates.
- Combined graph/phi landmarks currently put the phase-like onset around
  eta `0.32..0.35`.
- The first-derivative signal is supportive but noisier than raw phi(d=1):
  ref1 finite-difference runs can move the dphi threshold earlier than the
  graph/energy saturation threshold.
- The promoted ref10/n1024 run confirms a smoother transition window:
  - k=3 NMSTV: eta 0.30 -> 0.884, eta 0.35 -> 0.934.
  - phi(d=1) raw energy: eta 0.25 -> -0.1504, eta 0.30 -> -0.1545,
    eta 0.35 -> -0.1577, eta 0.40 -> -0.1593.
  - ref-level d phi/dd at d=1: eta 0.25 -> -0.0188, eta 0.30 -> -0.0253,
    eta 0.35 -> -0.0140, eta 0.40 -> -0.0114.
  - The most stable current interpretation is onset near eta `0.32..0.35`,
    with raw phi saturation closer to eta `0.38`.
- The small-d extension (`d=0.1..1.0`) gives a positive-curvature-mass signal:
  - `A_kappa`: eta 0.25 -> 0.541, eta 0.30 -> 0.542,
    eta 0.35 -> 0.611, eta 0.40 -> 0.681.
  - Mean minimum curvature remains negative for all eta; its mean location is
    around `d=0.67..0.86` across the four eta values.

## Resource Policy Used

- CPU target: <=35%.
- GPU target: <=25%.
- Current runs used CPU only and set `CUDA_VISIBLE_DEVICES=""`.
- Thread cap: 8 on a 32-core machine.

## Timing

- Single anchor smoke unit at n=128, d=1.0: 2.79 s SMC elapsed.
- 20-unit short d=1 anchor smoke at n=128: mean unit 4.41 s, max 7.52 s.
- Eta-specific 60-unit n=128/ref3 run: mean unit 4.01 s, max 5.49 s.
- Eta-specific 20-unit n=1024/ref1 run: mean unit 24.14 s, max 33.02 s.
- Dense eta 24-unit n=1024/ref1 run: mean unit 26.87 s, max 33.71 s.
- Promoted 160-unit n=1024/ref10 run: mean unit 25.95 s, max 35.78 s.
- Promoted small-d extension to 400 active units: mean unit 21.54 s,
  max 35.78 s; matching smoke first new low-d unit took about 16.5 s wall.
- A 4 eta x 5 radius x 30 ref n=1024 run is therefore likely around 10
  CPU-hours at the current 8-thread CPU-only setting, before overhead.

## Primary Outputs

- Graph proxy:
  - `raw_outputs/eta_sweep_pilot_cpu35_gpu0/summary_by_eta_k.csv`
  - `raw_outputs/eta_sweep_pilot_cpu35_gpu0/derivative_by_eta_k.csv`
  - `figures/eta_sweep_pilot_cpu35_gpu0/fig01_eta_knn_nmstv_by_k.png`
  - `figures/eta_sweep_pilot_cpu35_gpu0/fig02_eta_first_derivative_proxy.png`

- Anchor phi smoke:
  - `raw_outputs/eta_anchor_phi_short_d1_cpu35_gpu0/06_results_figures/eta_anchor_phi_by_eta_radius.csv`
  - `raw_outputs/eta_anchor_phi_short_d1_cpu35_gpu0/06_results_figures/eta_anchor_dphi_dd_by_eta_radius.csv`
  - `raw_outputs/eta_anchor_phi_short_d1_cpu35_gpu0/06_results_figures/fig02_eta_anchor_phi_energy_d1_zoom.png`
  - `raw_outputs/eta_anchor_phi_short_d1_cpu35_gpu0/06_results_figures/fig03_eta_anchor_delta_phi_energy_d1_zoom.png`

- Eta-specific reference search:
  - `raw_outputs/eta_reference_search_4eta_3ref_cpu35_gpu0/04_exact_reference_search/reference_index.csv`
  - `raw_outputs/eta_reference_search_dense_eta_ref1_cpu35_gpu0/04_exact_reference_search/reference_index.csv`

- Eta-specific phi:
  - `raw_outputs/eta_reference_phi_4eta_3ref_d1_n128_cpu35_gpu0/06_results_figures/eta_reference_phi_by_eta_radius.csv`
  - `raw_outputs/eta_reference_phi_4eta_1ref_d1_n1024_cpu35_gpu0/06_results_figures/eta_reference_phi_by_eta_radius.csv`
  - `raw_outputs/eta_reference_phi_dense_eta_ref1_d1_n1024_cpu35_gpu0/06_results_figures/eta_reference_phi_by_eta_radius.csv`
  - `raw_outputs/eta_reference_phi_promoted_4eta_10ref_d1_n1024_cpu35_gpu0/06_results_figures/eta_reference_phi_by_eta_radius.csv`
  - `raw_outputs/eta_reference_phi_promoted_4eta_10ref_d1_n1024_cpu35_gpu0/05_pool2_pm_sais_sampling/shell_summary_by_unit_with_phi_derivatives.csv`

- Positive curvature mass:
  - `figures/eta_positive_curvature_mass_small_d_n1024/fig01_eta_positive_curvature_mass_composite.png`
  - `figures/eta_positive_curvature_mass_small_d_n1024/eta_positive_curvature_mass_by_eta.csv`
  - `figures/eta_positive_curvature_mass_small_d_n1024/eta_positive_curvature_mass_by_ref.csv`
  - `figures/eta_positive_curvature_mass_small_d_n1024/eta_curvature_by_eta_ref_radius.csv`

- Phase summary:
  - `figures/eta_phase_summary_cpu35_gpu0/fig01_eta_phase_summary.png`
  - `figures/eta_phase_summary_n1024_ref1_cpu35_gpu0/fig01_eta_phase_summary.png`
  - `figures/eta_phase_summary_dense_eta_n1024_ref1_cpu35_gpu0/fig01_eta_phase_summary.png`
  - `figures/eta_phase_summary_dense_eta_n1024_ref1_cpu35_gpu0/eta_phase_landmarks.csv`
  - `figures/eta_phase_summary_promoted_10ref_n1024_cpu35_gpu0/fig01_eta_phase_summary.png`
  - `figures/eta_phase_summary_promoted_10ref_n1024_cpu35_gpu0/fig02_nmstv_vs_dphi_phase_plane.png`
  - `figures/eta_phase_summary_promoted_10ref_n1024_cpu35_gpu0/eta_phase_summary_table.csv`

## Remaining Formal Work

- The focused 10-ref/n1024 run has now been completed.
- A stronger final claim would need either more references or repeated
  independent n=1024 split seeds for the same references, because the strict
  split-logZ diagnostic can still fail.
- The small-d curvature pass is now available, but it uses finite differences
  on a `0.1` radius grid; denser small-d radii would be needed before treating
  the curvature extremum location as final.
- The strict split-logZ diagnostic still needs attention: n=1024 ref1 runs
  and the promoted ref10 run can still exceed the old 0.004 gate.
