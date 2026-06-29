# NMSTV Axis Diagnostic And Spin Comparison

## Main finding

The even odd / eta=0.05 inversion was caused by stale/intermediate NMSTV metadata in the older complexity-axis table.
When NMSTV is recomputed from the actual direct-run dataset labels used by the sampling/reference rows, eta=0.05 moves to the right of even odd.

- stale eta=0.05 NMSTV: `0.430574`
- corrected eta=0.05 NMSTV: `0.573493`
- corrected even odd NMSTV: `0.493286`
- corrected rule/eta/random trend: `very low tv (0.325) -> even odd (0.493) -> eta=0.05 (0.573) -> eta=0.15 (0.776) -> eta=0.25 (0.898) -> random label (0.986)`

The teacher NN rule is kept as a separate rule family and is not part of the monotone label-flip interpolation.

## Method notes

- Corrected MNIST complexity is graph TV NMSTV recomputed on the actual dataset paths in the direct run.
- The eta expectation uses the even odd graph-TV mean and independent label-flip formula `a=2 eta (1-eta)`, `TV_eta ~= TV_even*(1-2a)+a`.
- MNIST phase metric uses direct first derivatives from sampling and finite differences only for the second derivative.
- The error metric line is reference-pool test error, drawn on the MNIST phase panels.
- The 3NN comparison uses the previous 18 beta, 90 dataset, 30 reference dense spin tables.

## Figures

- `/home/bjyong/Complexity/local_project/03_dnn_mnist/06_eta_flip_phase_transition/figures/nmstv_axis_diagnostic_and_spin_comparison_direct30ref/fig01_nmstv_axis_diagnostic.png`
- `/home/bjyong/Complexity/local_project/03_dnn_mnist/06_eta_flip_phase_transition/figures/nmstv_axis_diagnostic_and_spin_comparison_direct30ref/fig02_mnist_direct_vs_3nn_spin_composite.png`
- `/home/bjyong/Complexity/local_project/03_dnn_mnist/06_eta_flip_phase_transition/figures/nmstv_axis_diagnostic_and_spin_comparison_direct30ref/fig03_normalized_phase_overlay_spin_mnist.png`

## Key tables

- `corrected_mnist_case_metrics.csv`
- `stale_vs_corrected_nmstv.csv`
- `eta_expected_nmstv_from_even_odd.csv`
- `spin_case_metrics.csv`
