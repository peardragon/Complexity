# 03_dnn_mnist

10×10 MNIST, train 512개. 100–20–20–1 tanh, P=2461.
조건별 dataset 10개 × reference 10개. r=.01–1.00, SMC 512 particles × 2 pools.

| 실험 | 조건 | 논문 |
|---|---|---|
| label_noise_sweep | 같은 이미지에서 eta=0,.05,.15,.25,.5 | Figs. 5–6 |
| digit_pairwise_complexity | 45쌍의 평균 C_MS 순위 중 1,5,…,45위 | Figs. 7–8 |

두 실험 모두 02와 같은 5단계: dataset → complexity → reference → sampling → entropy.
실행은 `src/*.py`, 보조 함수는 `src/utils/*.py`.

```bash
python 03_dnn_mnist/label_noise_sweep/01_dataset/src/make_dataset.py --dataset-index 0
python 03_dnn_mnist/label_noise_sweep/03_reference_search/src/reference_search.py --dataset-index 0
python 03_dnn_mnist/label_noise_sweep/04_sampling/src/sampling.py --dataset-index 0 --shard-index 0 --shard-count 5
```

실제 계산은 `--execute`. 기존 파일은 건너뜀. `--resume`는 생략 가능.
Digit-pair는 경로를 바꾸고 `--shard-count 12` 사용.
요약은 `--check-only`로 기존 compact 결과를 읽기 전용 검증하고,
`--execute`로 누락 파일만 생성하며, `--force`로 명시적으로 재계산한다.
세 요약 단계는 `--config PATH`와 `--output-dir DIR`를 지원한다. 설정 안의
입력 경로는 실험 root 기준, 출력 경로는 해당 stage 기준이며
`--output-dir`를 주면 설정된 파일의 basename만 그 디렉터리에 쓴다.

- `default.json`: 실제 실행 또는 요약에서 읽는 수치·조건·입출력 설정.
  Seed/RNG·정규화·미분 방식의 문자열은 계산 관례를 식별하는 값.
  별도 승인·promotion·설명문 일치 검사는 사용하지 않음.
- `objective.json`: 기본 loss (1,.01)에 beta=100을 반영한 (100,1).
- `resources.json`: CPU 최대 24 threads, GPU 최대 2개. GPU 선택은
  `CUDA_VISIBLE_DEVICES` 또는 `--device`로 실행 환경에서 지정한다.
- `frozen_pair_manifest.json`: 논문에서 사용한 12쌍과 순위.
- `summarized_outputs/r1_weighted_accuracy.json`: 기존 raw shard용 보존 정확도.
  새로 생성한 r=1 shard에는 terminal particle의
  `weighted_training_accuracy`가 들어가며 05는 한 condition 전체에 이 필드가
  있을 때만 이를 사용한다. condition 안에서 새 값과 보존 값을 섞지 않는다.

`raw_outputs/`는 대용량 재생성 입력이라 release Git에는 포함하지 않는다.
raw 자료가 별도로 제공되면 02/04/05의 `--execute --output-dir ...`로 compact
결과를 독립적으로 재생성할 수 있다. 기존 파일은 내용 hash가 아니라 파일명으로
건너뛰되, condition/dataset/reference/radius 중복·누락과 유한값·범위를 검사한다.
