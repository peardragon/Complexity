# Paper Figures

Main Figs. 1–9 for arXiv:2608.22361v1.

## Project Overview

- **Fig. 1**: Dataset-to-landscape schematic.
- **Fig. 2**: Perceptron benchmark.
- **Figs. 3–4**: Synthetic experiment.
- **Figs. 5–6**: MNIST label noise.
- **Figs. 7–8**: MNIST digit pairs.
- **Fig. 9**: Normalized radial response and corridor sketches.

## Playground

The executed [figure notebook](releases/rebuild_all_paper_figures.ipynb) contains a setup cell, one cell per figure, and an output audit. Edit each `conf` for layout, spacing, legends, and annotations.

All cell outputs and embedded images are retained. The notebook rebuilds figures, not the underlying training or sampling experiments.

## Workflow & Dependencies

```bash
python Figures/paper_figures/src/build_all.py
python Figures/paper_figures/src/build_all.py --check-only
python Figures/paper_figures/src/validate_release.py
```

- **build_all.py**
  - **Utils Dependencies**: `paper_style`, `notebook_runner`.
  - **Purpose**: Synchronize numerical inputs and rebuild missing figures with the notebook settings.
- **render_quantitative.py**
  - **Purpose**: Render the perceptron, synthetic, and MNIST numerical panels.
- **render_discussion.py**
  - **Purpose**: Render the empirical hardening panel and corridor sketches.
- **stage_static_assets.py**
  - **Purpose**: Export the supplied overview image.
- **build_mnist_umap_assets.py**
  - **Utils Dependencies**: `mnist_umap_assets`.
  - **Purpose**: Create missing endpoint datasets and UMAP panels with the original visual configuration. Existing PDFs skip before heavy imports.
- **validate_release.py**
  - **Purpose**: Read-only validation of numerical authorities, r=1 reconstruction, staged inputs, notebook images, and figure exports.

Missing output formats are generated individually. Existing companion files are preserved. Source-summary changes update the staged inputs and affected figures. `--force` explicitly redraws all figures.

## Configuration

- **figure_manifest.json**: Figure IDs, renderers, inputs, and outputs.
- **input_sources.json**: Source-to-staged input mapping.
- **../config/paper_figure_style.json**: Common typography and physical widths.
- **Notebook conf**: Per-figure settings used by the build.

## Results Storage

Inputs are in `figure_inputs/`; exports follow the figure manifest and include PDF, PNG, and SVG. Build records are in `receipts/`, and the inventory is in `summarized_outputs/`.

Raw files are not required to rebuild the supplied figures. If a MNIST panel is missing, the normal build bootstraps its dataset and UMAP visual input before synchronization. The retained PDFs are always preferred; `--check-only` never creates inputs. Fig. 1 is an authored schematic included with the release.
