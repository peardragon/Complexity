# Complexity-Axis Spin/MNIST Figures

This bundle freezes the current discussion figures and rebuilds the comparison on an empirical complexity axis.

## Complexity Convention

- Spin: `C_spin = knn_edge_disagreement_mean`, normalized within the spin beta sweep.
- MNIST: `C_mnist = NMSTV`, normalized within the current four-rule + four-eta panel.
- Larger normalized `C` means more locally disordered/complex labels. This avoids comparing raw `beta` and `eta` directions directly.

## Frozen Inputs

- Frozen file count: `18`
- Frozen directory: `/home/bjyong/Complexity/local_project/03_dnn_mnist/06_eta_flip_phase_transition/figures/complexity_axis_spin_mnist_30ref_eta0p02_0p05_0p15_0p25/00_frozen_current_figures`

## Spin Complexity Range

- beta range: `0.05` to `0.39`
- spin C proxy range: `0.0712854` to `0.478021`

## MNIST Complexity Table

| label | source | NMSTV | C_norm | A_kappa(SG21 mean curve) | phi_E(d=1) |
| ----- | ------ | ----- | ------ | ------------------------ | ---------- |
| adv very low tv | advanced | 0.32457 | 0.000 | 0.0455513 | -0.0564072 |
| flip eta 0.02 | flip | 0.356969 | 0.049 | 0.12339 | -0.0843992 |
| flip eta 0.05 | flip | 0.430574 | 0.160 | 0.178321 | -0.0987867 |
| adv even odd | advanced | 0.493286 | 0.255 | 0.097941 | -0.0741446 |
| flip eta 0.15 | flip | 0.655129 | 0.500 | 0.385784 | -0.134355 |
| adv teacher nn | advanced | 0.684377 | 0.544 | 0.110617 | -0.105078 |
| flip eta 0.25 | flip | 0.811137 | 0.736 | 0.52539 | -0.149625 |
| adv random | advanced | 0.985559 | 1.000 | 0.658162 | -0.160985 |

## Outputs

- `fig01_mnist_phi_energy_by_complexity_axis.png`
- `fig02_complexity_axis_phase_metrics.png`
- `fig03_complexity_axis_Akappa_separate_scales.png`
- `spin_complexity_axis_metrics.csv`
- `mnist_complexity_axis_metrics.csv`
- `00_frozen_current_figures/frozen_current_figures_manifest.csv`
