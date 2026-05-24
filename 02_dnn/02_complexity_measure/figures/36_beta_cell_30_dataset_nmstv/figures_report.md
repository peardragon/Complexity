# Complexity measure figures report: 36_beta_cell_30_dataset_nmstv

이 폴더는 complexity measure stage의 active figure만 보관한다. root `figures/*.png` 중복본은 backup으로 이동했고, run별 하위 폴더가 canonical figure 위치이다.

## 입력 데이터

입력: `02_dnn/02_complexity_measure/raw_outputs/36_beta_cell_30_dataset_nmstv/summary_tables/`의 complexity summary CSV/JSON.

## 그림 설명

- `beta_series_grid.png`: complexity 측정에 사용된 beta series dataset의 대표 격자 overview이다.
- `beta_series_grid_scatter.png`: complexity 측정 대상 dataset의 point cloud를 beta 순서대로 보여준다.
- `nmstv_scale_crossing_count_beta_series.png`: beta별 nMSTV scale crossing 개수를 요약해 multiscale ordering 변화량을 보여준다.
- `nmstv_scale_crossing_robust_beta_series.png`: scale crossing이 robust하게 유지되는 beta 구간을 확인하는 그림이다.
- `nmstv_scale_heatmap_beta_series.png`: beta와 scale 축에서 nMSTV 값을 heatmap으로 보여준다.
- `nmstv_scale_rank_heatmap_beta_series.png`: beta와 scale 축에서 nMSTV rank/order 변화를 heatmap으로 보여준다.
- `nmstv_scale_span_vs_beta.png`: beta별 scale span에 따른 nMSTV 변동 폭을 요약한다.
- `nmstv_vs_beta.png`: cell-level aggregate nMSTV가 beta에 따라 어떻게 변하는지 보여준다.
- `nmstv_vs_beta_raw.png`: dataset-level raw nMSTV 분포를 beta별로 보여준다.
- `nmstv_vs_scale_beta_series.png`: 각 beta series에서 scale별 nMSTV 곡선을 비교한다.
- `rho_scale_heatmap_beta_series.png`: beta와 scale 축에서 rho summary를 heatmap으로 보여준다.
- `rho_vs_scale_beta_series.png`: 각 beta series에서 scale별 rho 변화를 비교한다.
