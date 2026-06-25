# High-Beta Small-d CI Status

Scope: `dphi_energy/dd` 95% confidence interval half-width, high beta `beta >= 0.29`.

## 90 Dataset Status

- aggregate timestamp: `2026-06-14T18:18:48+09:00`
- sampling progress: `11794275/12150000` (`97.0722%`)
- failed units: `0`
- 90-dataset CI is partial: completed shard CSVs used `26/32`

## Mean CI Half-Width

| run | d range | cells | median datasets | mean CI half-width | median CI half-width |
| --- | --- | ---: | ---: | ---: | ---: |
| 30 dataset | 0.01_to_0.10 | 60 | 30 | 6.36832e-05 | 6.90656e-05 |
| 30 dataset | 0.01_to_0.30 | 180 | 30 | 0.000108966 | 0.000110647 |
| 60 dataset | 0.01_to_0.10 | 60 | 60 | 3.81838e-05 | 4.10066e-05 |
| 60 dataset | 0.01_to_0.30 | 180 | 60 | 6.67359e-05 | 6.82117e-05 |
| 90 dataset partial | 0.01_to_0.10 | 60 | 90 | 3.05077e-05 | 3.35843e-05 |
| 90 dataset partial | 0.01_to_0.30 | 180 | 90 | 5.30337e-05 | 5.44863e-05 |

## Relative Change

- d=0.01-0.10: 60/30 mean CI ratio `0.600`, 90partial/60 ratio `0.799`.
- d=0.01-0.30: 60/30 mean CI ratio `0.612`, 90partial/60 ratio `0.795`.

## Outputs

- detail_csv: `02_dnn/05_proxy_local_entropy/figures/dphi_dd_30_60_90_ci_small_d/dphi_dd_high_beta_small_d_ci_detail.csv`
- summary_csv: `02_dnn/05_proxy_local_entropy/figures/dphi_dd_30_60_90_ci_small_d/dphi_dd_high_beta_small_d_ci_summary.csv`
- small_d_0p01_0p30_plot: `02_dnn/05_proxy_local_entropy/figures/dphi_dd_30_60_90_ci_small_d/dphi_energy_dr_ci95_high_beta_small_d_0p01_0p30.png`
- small_d_0p01_0p10_plot: `02_dnn/05_proxy_local_entropy/figures/dphi_dd_30_60_90_ci_small_d/dphi_energy_dr_ci95_high_beta_small_d_0p01_0p10.png`
- mean_bar_plot: `02_dnn/05_proxy_local_entropy/figures/dphi_dd_30_60_90_ci_small_d/dphi_energy_dr_ci95_high_beta_small_d_mean_bar.png`
- report: `02_dnn/05_proxy_local_entropy/figures/dphi_dd_30_60_90_ci_small_d/high_beta_small_d_ci_report.md`
