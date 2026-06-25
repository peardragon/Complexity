# Final Acceptance And High-Beta Comparison

Generated: `2026-06-15T11:58:44+09:00`

## Acceptance

- claim: `pass`
- failed units: `0`
- sampling rows: `12150000/12150000`
- proxy rows: absolute `4500`, delta `4500`, dphi `4500`; expected `4500` each
- failed beta-radius cells: `0`
- beta-radius acceptance table: `02_dnn/04_sampling/raw_outputs/shell_pool/18_beta_cell_90_dataset_30_reference/d_0.01_to_2.50_dense/summary_tables/final_acceptance_beta_radius.csv`

## High-Beta Comparison

- high-beta threshold: `beta >= 0.29`
- 30-dataset baseline significant radius fraction: `0.5720`
- 90-dataset target significant radius fraction: `0.3600`
- baseline median slope: `0.0014453307`
- target median slope: `0.00075389586`
- conclusion: `high-beta curves show a localized trend, but not a radius-broad one`

The CI check uses per-dataset means of the full energy term `logZ_inf_full / P`
within each beta-radius cell, then fits a high-beta linear slope per radius.
