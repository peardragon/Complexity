# Eta Flip Pilot Report

## Run

- run_name: `eta_sweep_pilot_cpu35_gpu0`
- base dataset: `/home/bjyong/Complexity/windows_project/02_dnn/08_mnist/runs/final/single_dataset_10x10_box_n_train_512_60ref_allrule_sparse_0p010_to_2p500/01_dataset_prepare/raw_datasets/split_000/real_even_odd/dataset.npz`
- eta grid: `[0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]`
- replicates: `20` with nested flip uniforms
- kNN graph k values: `[3, 5, 8, 16, 32]`
- resource policy: CPU cap target 35%, GPU cap target 25%; this run used CPU only with thread_limit=8
- elapsed seconds: `2.67`
- compact label store: `/home/bjyong/Complexity/local_project/03_dnn_mnist/06_eta_flip_phase_transition/raw_outputs/eta_sweep_pilot_cpu35_gpu0/eta_nested_label_store.npz`

## Main Read

This pilot treats eta as a label-noise knob on the fixed MNIST even/odd inputs.
For the graph-TV proxy, independent label flips have a closed-form expectation:

`q_eta = q0 + 2 * eta * (1 - eta) * (1 - 2*q0)`

where `q0` is the base edge-disagreement rate. Therefore the first derivative
is largest near eta=0 and decays toward eta=0.5 when q0<0.5. In this proxy
alone, an interior positive-curvature transition is not expected; if a
random-like transition appears in phi(d), it should come from the DNN
sampling/reference geometry rather than from graph-TV algebra alone.

## k=3 Landmarks

- NMSTV at eta=0: `0.3001`
- NMSTV at eta=0.5: `1.0001`
- eta crossing NMSTV>=0.90: `0.350`
- peak first derivative eta: `0.000`
- expected d2 cut / d eta2 at eta=0, k=3: `-2.7997`

## Output Tables

- `summary_by_eta_k.csv`
- `replicate_metrics.csv`
- `derivative_by_eta_k.csv`
- `eta_transition_candidates.csv`
- `expected_independent_flip_curve.csv`

## Output Figures

- `fig01_eta_knn_nmstv_by_k.png`
- `fig02_eta_first_derivative_proxy.png`
- `fig03_eta_transition_landmarks.png`
- `fig04_eta_nested_label_examples.png`
- `fig05_reference_spin_positive_curvature_mass.png` when the reference file is available
