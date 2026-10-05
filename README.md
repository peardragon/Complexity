# Complexity: Dataset Complexity and Finite-Distance Loss Geometry

This project studies how dataset complexity changes the loss landscape around trained neural networks. It includes a perceptron benchmark, controlled synthetic data, two MNIST experiments, radial analysis, and the paper figures.

Paper: [Dataset Complexity Shapes Finite-Distance Loss Geometry in Neural Networks](https://arxiv.org/abs/2608.22361v1), Jaeyong Bae and Hawoong Jeong.

## Project Overview

- **Perceptron benchmark**: Replica-symmetric calculation and finite-size shell SMC.
- **Synthetic experiment**: Label organization controlled by Kawasaki dynamics.
- **MNIST label noise**: Paired inputs with nested label flips.
- **MNIST digit pairs**: Twelve tasks selected from the mean complexity ranking of all 45 pairs.
- **Radial analysis**: Local entropy, direct radial derivative, total radial variation, and normalized hardening.
- **Functional check**: Particle-weighted training accuracy at r=1.
- **Discussion**: Symmetric reference-tilt check, random-label reentrance, and corridor sketches.

Processed numerical results and the executed figure notebook are included. Large datasets, reference pools, and sampling shards are excluded.

The current perceptron input commands validate the original fixed pools; they do not generate those pools. These inputs must be supplied separately for new theory sampling. MNIST visualizations use the retained UMAP panels, and Fig. 1 uses the supplied schematic image.

## Installation

Install the tested dependencies using Python 3.12:

```bash
python -m pip install -r requirements.txt
```

The release was checked on Linux. CUDA-enabled PyTorch is required for GPU sampling. LaTeX and Latin Modern are required to redraw the notebook with its original typography. Numba is included for the production replica-saddle calculation.

## Playground

The [paper figure notebook](Figures/paper_figures/releases/rebuild_all_paper_figures.ipynb) rebuilds the nine main figures from the included inputs. It contains:

1. **Overview**: Dataset-to-landscape schematic.
2. **Perceptron benchmark**: Replica-symmetric and finite-size curves.
3. **Synthetic experiment**: Dataset organization and local entropy.
4. **MNIST experiments**: Label noise and digit-pair comparisons.
5. **Discussion**: Normalized radial response and corridor sketches.
6. **Output checks**: Export dimensions and figure inventory.

Figure settings are edited in each cell's `conf`. The notebook includes executed outputs and embedded images. It does not run training or SMC.

## Directory Structure

```text
Complexity/
├── 01_theory/
│   ├── 01_theory_analytic/          # Replica-symmetric calculation
│   └── 02_theory_sampling/          # Fixed-input SMC and nested aggregation
├── 02_dnn_synthetic/
│   ├── 01_dataset/                 # Synthetic data generation
│   ├── 02_complexity_measure/      # Multiscale dataset complexity
│   ├── 03_reference_search/        # Zero-error reference search
│   ├── 04_sampling/                # Shell SMC and sampling QC
│   └── 05_proxy_local_entropy/     # Profiles, radial variation, r=1 accuracy
├── 03_dnn_mnist/
│   ├── label_noise_sweep/          # Same five stages
│   └── digit_pairwise_complexity/  # Same five stages
├── 04_discussion/
│   ├── 01_frozen_inputs/           # Input inventory
│   ├── 02_normalized_radial_geometry/
│   ├── 03_antipodal_geometry/      # Appendix D
│   └── 04_random_label_reentrance/
└── Figures/
    ├── config/                    # Common figure style
    ├── paper_figures/             # Figs. 1–9 and executed notebook
    └── 04_discussion/             # Fig. 9 copy and companion plots
```

Each numerical stage uses `config/default.json`, `src/*.py`, and `src/utils/*.py`. Additional settings are described in the experiment READMEs:

- **objective.json**: Loss coefficients and numerical conventions.
- **resources.json**: CPU/GPU limits and deterministic backend settings.
- **frozen_pair_manifest.json**: Selected digit pairs and their ranks.

## Workflow & Dependencies

### 1. Theory

- **theory_full_rs.py**
  - **Utils Dependencies**: `rs_refinement_core`.
  - **Purpose**: Calculate the replica-symmetric perceptron curve.
- **sampling.py**
  - **Utils Dependencies**: `schedule`, `smc`, `perceptron`.
  - **Purpose**: Run shell SMC using the supplied dataset/reference pools.
- **make_summarized_outputs.py**
  - **Utils Dependencies**: `aggregate`.
  - **Purpose**: Average references within datasets, then calculate dataset means and SEM.

The full-shell geometry follows the squared-distance delta convention, with coefficient `(N-2)/N`.

### 2. DNN Data Generation & Reference Search

Synthetic and both MNIST experiments use the same five-stage workflow:

`dataset → complexity → reference → sampling → local entropy`

- **01_dataset/src/make_dataset.py**: Generate the selected datasets.
- **02_complexity_measure/src/make_summarized_outputs.py**: Calculate C_MS.
- **03_reference_search/src/reference_search.py**: Retain zero-training-error references.
- **04_sampling/src/sampling.py**: Sample the loss-weighted shells.
- **05_proxy_local_entropy/src/make_summarized_outputs.py**: Aggregate profiles and condition metrics.

These entry points use dry-run/check modes by default. `--execute` runs the requested operation. Existing outputs are skipped by filename. Summary scripts accept `--config`, `--output-dir`, and `--check-only`; `--force` explicitly rebuilds outputs.

### 3. Training Accuracy at r=1

- **05_proxy_local_entropy/src/make_r1_accuracy.py**
  - **Utils Dependencies**: `r1_accuracy`.
  - **Purpose**: Rebuild the accuracy JSON from the original saved reference-level observations.
- **05_proxy_local_entropy/frozen_inputs/**
  - **Contents**: Compact split accuracies, split-logZ mixture weights, and reference accuracies.
  - **Aggregation**: Reference mean within each dataset, then dataset mean and sample SEM.

No GPU or resampling is needed for this reconstruction. The original capture and aggregation sources are listed in each `frozen_inputs/README.md`.

New sampling records `weighted_training_accuracy` at r=1. Legacy shards use the saved JSON, or the included capture table if the JSON is absent. Fresh and archived values are not mixed within one condition.

### 4. Discussion

- **04_discussion/src/run_all.py**
  - **Utils Dependencies**: `discussion_pipeline`, `antipodal_summary`.
  - **Purpose**: Generate normalized radial profiles, symmetric-response tables, and reentrant intervals.
- **03_antipodal_geometry/src/evaluate_antipodal.py**
  - **Purpose**: Evaluate the two-sided response on retained MNIST references.

Fig. 9(b,c) are schematic interpretations, not measured loss slices or Hessian spectra. The original antipodal NPZ files are not distributed; compact tables are included for the published analysis.

### 5. Figures & Checks

```bash
python Figures/paper_figures/src/validate_release.py
python Figures/paper_figures/src/build_all.py
python Figures/04_discussion/src/make_figures.py
```

The build synchronizes the included source summaries with the figure inputs. Missing output formats are generated without replacing completed companion files. Use `build_all.py --force` to redraw with the notebook settings.

## Utilities Function Reference

### Numerical Modules

- **rs_refinement_core.py**: `solve_q_ref()`, `RSRefinementCore.evaluate()`, and relative saddle curves.
- **dataset.py**: `mutual_knn_graph()`, `kawasaki_labels()`, and `generate_arrays()`.
- **datasets.py**: MNIST loading, box averaging, standardization, and task construction.
- **reference.py / reference_training.py**: Reference initialization, optimizer updates, and zero-error selection.
- **smc.py / shell_smc.py**: Adaptive tempering, vMF mutation, split combination, and direct radial scores.
- **r1_accuracy.py**: Saved-capture validation and reference-to-dataset accuracy aggregation.

### Analysis & Figure Modules

- **aggregate.py**: Theory shard loading and nested mean/SEM calculation.
- **discussion_pipeline.py**: Normalized radial response and reentrant intervals.
- **antipodal_summary.py**: Dataset-block aggregation of the symmetric response.
- **paper_style.py**: Typography, physical widths, and PDF/PNG/SVG export.
- **notebook_api.py**: Editable notebook adapters for `fig01()` through `fig09()`.
- **notebook_runner.py**: Rebuild selected figures with the notebook settings.

## Results Storage

- Numerical summaries are stored in each stage's `summarized_outputs/`.
- Compact r=1 observations are stored in `05_proxy_local_entropy/frozen_inputs/`.
- Figure inputs are stored in `Figures/paper_figures/figure_inputs/`.
- Main and companion figures are saved as PDF, PNG, and SVG.
- The executed notebook is kept in `Figures/paper_figures/releases/`.
- `RELEASE_MANIFEST.json` records the public files and tested package versions.

Large raw files, caches, backups, and private working settings are excluded from Git. Saved-raw reaggregation and the raw-excluded figure workflow were checked; a complete new production GPU experiment was not rerun during release preparation.
