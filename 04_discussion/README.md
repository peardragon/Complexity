# Discussion: Finite-Distance Geometry

Analysis of the radial response, the symmetric reference-tilt check, and random-label reentrance in [arXiv:2608.22361v1](https://arxiv.org/html/2608.22361v1).

## Project Overview

- **01_frozen_inputs**: Inventory of the numerical inputs.
- **02_normalized_radial_geometry**: h(r)=-g_E(r)/r, peaks, and turning scales; Discussion IV.1 and Fig. 9(a).
- **03_antipodal_geometry**: Clean/random two-sided response K_r; the final label-noise Results paragraph and Appendix D.
- **04_random_label_reentrance**: Negative–positive–negative intervals and zero crossings; Fig. 6(b) and Discussion IV.3.

Fig. 9(b,c) are corridor sketches. The fixed-Hessian argument is an analytic derivation in the paper, not a Hessian-spectrum experiment in this repository.

## Workflow & Dependencies

```bash
python 04_discussion/src/run_all.py
python 04_discussion/src/validate_all.py
```

- **run_all.py**
  - **Utils Dependencies**: `discussion_pipeline`, `antipodal_summary`.
  - **Purpose**: Build the four stages from the upstream profiles and saved antipodal evaluations.
- **validate_all.py**
  - **Purpose**: Read-only checks of coordinates, finite values, and the published numerical relationships.

Existing outputs are skipped individually. Use `--force` to rebuild. Each stage has `config/default.json`; helpers are in `src/utils/`.

For a separate comparison:

```bash
python 04_discussion/src/run_all.py --output-dir /path/to/comparison
python 04_discussion/src/validate_all.py --output-dir /path/to/comparison
```

## Appendix D Evaluation

For a raw-free checkout, first create the ten MNIST datasets and the clean/random reference ensembles:

```bash
for index in {0..9}; do
  python 03_dnn_mnist/label_noise_sweep/01_dataset/src/make_dataset.py --dataset-index "$index" --execute
  for condition in noise_eta_0p00 noise_eta_0p50; do
    python 03_dnn_mnist/label_noise_sweep/03_reference_search/src/reference_search.py --dataset-index "$index" --condition "$condition" --device cuda:0 --execute
  done
done
for condition in noise_eta_0p00 noise_eta_0p50; do
  python 04_discussion/03_antipodal_geometry/src/evaluate_antipodal.py --condition "$condition" --device cuda:0 --execute
done
python 04_discussion/03_antipodal_geometry/src/make_summarized_outputs.py --force
```

Run these commands from the repository root with Bash. Select an available device instead of `cuda:0` if needed. MNIST is fetched from public OpenML when its source cache is absent. Existing dataset and reference outputs are reused by filename. No shell SMC is needed for this evaluation.

The evaluation uses ten datasets, ten references, 256 direction pairs, radii {.01,.03,.07,.10,.12,.20,.40}, and seed=2026082201. It evaluates the retained references; it does not train or run SMC.

The default Discussion build uses the included tables and does not run this fresh evaluation. `--force` explicitly replaces summaries with the newly evaluated results; new training realizations need not reproduce the saved paper values exactly.

Objective scale, parameter count, radii, seed/stride, directions, and chunks are read from the stage config. `--device` selects the evaluation device. Evaluation and summary scripts support `--output-dir` and `--force`.

If evaluation NPZ files are written elsewhere, pass their directory to the summary using `--raw-dir`. Both original NPZ files trigger raw aggregation; both absent trigger the included compact-table path. An incomplete NPZ pair is rejected.

## Results Storage

Normalized profiles and reentrant intervals are stored in each stage's `summarized_outputs/`. The four `matched_antipodal_*` tables preserve the original analysis schema.

Original antipodal NPZ files are excluded from Git. Compact inputs are included in `03_antipodal_geometry/frozen_inputs/`. Discussion plots are generated under `Figures/04_discussion/`.
