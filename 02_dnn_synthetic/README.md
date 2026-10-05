# 02_dnn_synthetic

Synthetic DNN. Figs. 3–4, Appendix A/C.
Beta=0.05,0.07,…,0.39. 조건별 dataset 60개 × reference 10개.
2–48–48–1 tanh, r=0.01–2.50, SMC 512 particles × 2 pools.

| 단계 | 내용 |
|---|---|
| 01_dataset | Dataset 생성 |
| 02_complexity_measure | C_MS 계산 |
| 03_reference_search | Zero-error reference 탐색 |
| 04_sampling | Shell SMC와 sampling QC |
| 05_proxy_local_entropy | Energetic profile, A_TV, r=1 accuracy 요약 |

실행은 `src/*.py`, 보조 함수는 `src/utils/*.py`.

```bash
python 02_dnn_synthetic/01_dataset/src/make_dataset.py --start 0 --stop 1
python 02_dnn_synthetic/03_reference_search/src/reference_search.py --start 0 --stop 1
python 02_dnn_synthetic/04_sampling/src/sampling.py --dataset-job 0 --shell-pass odd
```

실제 계산은 `--execute`. 기존 파일은 파일명으로 건너뜀.
요약 스크립트는 `--check-only`로 확인, `--execute`로 누락분 생성, `--force`로 재계산.

- `default.json`: 실험 조건. 01/03/04는 같은 protocol, 02/05는 요약 설정.
  Seed/RNG·정규화·미분 방식의 문자열은 계산 관례를 식별하는 값.
- `objective.json`: 기본 loss (gamma,lambda)=(1,.01)에 beta=100을 반영한 (100,1).
- `resources.json`: CPU 최대 24 threads, GPU 최대 2개. GPU 선택은
  `CUDA_VISIBLE_DEVICES` 또는 실행의 `--device`로 지정. Worker당 BLAS thread 1.
- `summarized_outputs/r1_weighted_accuracy.json`: 기존 raw용 보존 정확도.
  새 sampling은 r=1 terminal particle의 `weighted_training_accuracy`를 저장.
  05는 condition 전체의 새 값이 있으면 사용하고, 없으면 archival 입력을 표시함.

요약은 `--config PATH`, `--output-dir DIR`, `--raw-root DIR` 지원.
Raw가 별도로 있으면 `--execute --output-dir <dir>`로 독립 재생성.
Raw dataset·reference·shard는 배포 제외. 요약 CSV와 작은 그림 입력은 포함.
코드·환경 hash는 식별 기록이며 기존 결과 재사용을 막는 조건이 아님.

r=1 JSON 재생성:

```bash
python 02_dnn_synthetic/05_proxy_local_entropy/src/make_r1_accuracy.py --check-only
python 02_dnn_synthetic/05_proxy_local_entropy/src/make_r1_accuracy.py --execute
```

05/frozen_inputs의 원래 10,800개 reference 관측을 재집계. GPU·SMC 실행 없음.
JSON이 없으면 05 요약에서도 이 표를 사용. 원래 계산·집계 코드 위치는 frozen_inputs/README.md.
