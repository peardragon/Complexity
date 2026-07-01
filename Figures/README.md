# Complexity Figures

This folder is a central, reproducible figure bundle. It is rebuilt from the current stage-level figure galleries and summarized-output figure builders.

Top-level PNG duplicates and stale legacy groups are intentionally removed by `rebuild_figures.py`.

## Groups

- `dnn_mnist/`: 43 figures.
- `dnn_synthetic/`: 27 figures.
- `theory/`: 7 figures.

## Sections

- `dnn_mnist/`: 43 figures.
- `dnn_synthetic/`: 27 figures.
- `theory/01_theory_analytic/`: 1 figures.
- `theory/02_theory_sampling/`: 5 figures.
- `theory/summary/`: 1 figures.

## Rebuild

Run `python Figures/rebuild_figures.py` from the project root.

The script first refreshes the stage galleries under `01_theory/figures`, `02_dnn_synthetic/figures`, and `03_dnn_mnist/figures`, then recopies current PNG outputs into grouped central folders.

`manifest.csv` records the grouped output, source image, and source summarized inputs when a stage manifest exposes them.

`rebuild_all_figures_from_summaries.ipynb` is an executable one-PNG-per-cell notebook. Each figure cell refreshes one source PNG, copies that single image into `Figures/`, prints the source/output paths, and displays that one image inline.
