# Proxy 3D Landscape Site Validation

Validation date: 2026-06-04

Server root:

`D:/Complexity/02_dnn/07_proxy_3D_landscape_for_visualize`

Active URL:

`http://127.0.0.1:8765/figures/proxy_landscape_v9_reviewed_site/index.html`

## Checks

- `node --check src/app.js`: pass.
- Browser page load: pass.
- Browser console errors/warnings: none observed.
- Copied target measures loaded from `raw_outputs/reference_atlas_target_measures`.
- Beta 0.23 interaction: pass. The site reports `B_q90` residual `-2.38` target SEM and marks that beta as a conservative read.
- Fixed proxy-loss scale: pass. The 3D surface z axis and color scale are fixed from the full 18-beta sweep range `[-51.01300811767578, 15.409536361694336]`.
- Surface markers and frame: pass. Reference-well / mean-center markers were reduced, and latent x/y scene ranges were widened beyond the source grid `[-4.2, 4.2]`.
- Standalone export: pass. `figures/proxy_landscape_v9_reviewed_standalone/proxy_3d_landscape_only.html` is a single self-contained HTML file with embedded Plotly and embedded proxy payload. Browser verification over local HTTP rendered the 3D canvas and preserved the camera/sweep behavior after switching to Top camera and beta `0.23`.

## Interpretation Note

The meaningful addition over the reviewed v9 static site is not a new unvalidated
surface formula. The visual proxy remains tied to the reviewed full-curve fit,
while the browser view now exposes target-SEM-scaled residuals and beta-wise
descriptor trajectories so the 3D surface is read in the context of the copied
target measures.
