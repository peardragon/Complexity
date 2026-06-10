# Proxy Local Entropy Figures

The active analysis figure is the interactive dashboard built from the retained
summary tables. Static PNG/CSV/report clutter from older figure passes is not
part of the active 05-stage figure contract.

- `local_entropy_dashboard.html`: combined 30-dataset and 60-dataset dashboard.
- `18_beta_cell_30_dataset_30_reference/d_0.01_to_2.50_dense/local_entropy_dashboard.html`: 30-dataset view.
- `18_beta_cell_60_dataset_30_reference/d_0.01_to_2.50_dense/local_entropy_dashboard.html`: 60-dataset view.
- `_assets/plotly-2.30.0.min.js`: local Plotly bundle used by the dashboards.

Use the dashboard controls for run selection, metric family, metric, beta
selection, radius window, linear/log radius scale, signed-log value transform,
and line/heatmap view.

Rebuild from the repository root with:

```powershell
python 02_dnn/05_proxy_local_entropy/src/build_interactive_dashboard.py --repo-root .
```
