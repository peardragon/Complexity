# Proxy local entropy figures report: d_0.01_to_2.50_dense

이 폴더는 proxy local entropy stage의 active figure를 보관한다. figure 입력은 같은 range의 05 raw_outputs proxy summary table이며, accuracy phase 계열만 full recovery 시 04 sampling sample payload NPZ를 다시 읽어 만든 accuracy quantile table이 추가로 필요하다.

## 입력 데이터

기본 입력: `02_dnn/05_proxy_local_entropy/raw_outputs/9_beta_cell_10_dataset_10_reference/d_0.01_to_2.50_dense/summary_tables/`.

원천 입력: `02_dnn/04_sampling/raw_outputs/shell_pool/9_beta_cell_10_dataset_10_reference/d_0.01_to_2.50_dense/summary_tables/` 및 필요 시 `sample_payloads/`.

## 그림 설명

- `accuracy_phase_map_q050.png`: q별 weighted accuracy cutoff phase map이다. 재생성에는 `accuracy_q_by_beta_radius.csv`가 필요하며, 이 table은 `make_proxy_tables.py --include-accuracy`로 sample payload NPZ를 다시 읽어 만든다.
- `accuracy_phase_map_q090.png`: q별 weighted accuracy cutoff phase map이다. 재생성에는 `accuracy_q_by_beta_radius.csv`가 필요하며, 이 table은 `make_proxy_tables.py --include-accuracy`로 sample payload NPZ를 다시 읽어 만든다.
- `accuracy_phase_map_q099.png`: q별 weighted accuracy cutoff phase map이다. 재생성에는 `accuracy_q_by_beta_radius.csv`가 필요하며, 이 table은 `make_proxy_tables.py --include-accuracy`로 sample payload NPZ를 다시 읽어 만든다.
- `accuracy_phase_maps_q050_q090_q099.png`: q=0.50, 0.90, 0.99 accuracy phase map을 한 번에 비교하는 panel figure이다. 재생성에는 accuracy quantile recovery가 필요하다.
- `delta_phi_by_distance.png`: `delta_phi_by_beta_radius.csv`에서 baseline radius 대비 delta phi를 distance별로 보여준다.
- `delta_phi_energy_term_by_distance.png`: `delta_phi_by_beta_radius.csv`의 energy term delta를 distance별로 보여준다.
- `delta_phi_entropic_term_by_distance.png`: `delta_phi_by_beta_radius.csv`의 entropic/area term delta를 distance별로 보여준다.
- `dphi_energy_dr_by_distance.png`: `dphi_dr_by_beta_radius.csv`의 energy term radial derivative를 distance별로 보여준다.
- `dphi_entropic_dr_by_distance.png`: `dphi_dr_by_beta_radius.csv`의 entropic term radial derivative를 distance별로 보여준다.
- `dphi_full_dr_by_distance.png`: `dphi_dr_by_beta_radius.csv`의 full phi radial derivative를 distance별로 보여준다.
- `fig_acc_phase_claim_map.png`: accuracy phase claim gate를 반영한 promoted alias figure이다. 재생성에는 accuracy quantile table이 필요하다.
- `fig_acc_phase_contour.png`: accuracy phase contour promoted alias figure이다. 재생성에는 accuracy quantile table이 필요하다.
- `fig_acc_phase_map.png`: accuracy phase map의 promoted alias figure이다. 재생성에는 accuracy quantile table이 필요하다.
- `fig_hq_claim_map.png`: `hq_by_beta_radius.csv` 기반 H threshold claim gate promoted alias figure이다.
- `fig_hq_phase_contour.png`: `hq_by_beta_radius.csv` 기반 H threshold contour promoted alias figure이다.
- `fig_hq_phase_map.png`: `hq_by_beta_radius.csv` 기반 H threshold phase map의 promoted alias figure이다.
- `hq_phase_map_q050.png`: raw_outputs의 `hq_by_beta_radius.csv`를 사용한 q별 H threshold phase map이다.
- `hq_phase_map_q090.png`: raw_outputs의 `hq_by_beta_radius.csv`를 사용한 q별 H threshold phase map이다.
- `hq_phase_map_q099.png`: raw_outputs의 `hq_by_beta_radius.csv`를 사용한 q별 H threshold phase map이다.
- `hq_phase_maps_q050_q090_q099.png`: q=0.50, 0.90, 0.99 H threshold phase map을 한 번에 비교하는 panel figure이다.
- `phi_by_distance.png`: raw_outputs의 `absolute_phi_by_beta_radius.csv`에서 distance별 full phi를 그린 핵심 proxy curve이다.
- `phi_energy_term_by_distance.png`: `absolute_phi_by_beta_radius.csv`의 energy term contribution을 distance별로 분리해서 보여준다.
- `phi_entropic_term_by_distance.png`: `absolute_phi_by_beta_radius.csv`의 geometric/entropic area term을 distance별로 보여준다.
