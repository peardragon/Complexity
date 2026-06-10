# Proxy 3D Landscape Visualization

This optional stage packages a browser visualization fitted to the active
18-beta/30-dataset reference-atlas diagnostics.

## Purpose

The 3D surface is a visual proxy, not a literal embedding of the full DNN
solution space. It should be interpreted together with the target metrics from
stage 06.

## Active Inputs

Copied reference-atlas target measures:

`raw_outputs/reference_atlas_target_measures/18_beta_cell_30_dataset_30_reference/d_0.01_to_2.50_dense/`

Reviewed proxy payload:

`raw_outputs/proxy_landscape_v9_reviewed_fit/`

## Active Figure Site

Serve the stage root and open:

`figures/proxy_landscape_v9_reviewed_site/index.html`

The site loads data from `raw_outputs/`, which is intentional: copied target
measures remain the source of truth, and the site is a presentation layer over
those retained outputs.

## Provenance

See `QC/provenance.json` and `QC/run_report.md` for copied source paths,
payload identity, and validation notes.
