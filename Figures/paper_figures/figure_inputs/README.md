# Figure inputs

그림 생성에 필요한 작은 입력만 보관.
CSV 요약, synthetic 예시 NPZ 4개, Fig. 1 PNG, MNIST UMAP PDF 4개.

경로 목록은 `../config/input_sources.json`.
기본 build는 파일 존재만 확인. `--refresh-inputs`로 입력 갱신.
Raw dataset 전체, reference, SMC shard는 포함하지 않음.
NPZ 4개는 그림 재현에 필요하므로 Git ignore에서 예외 처리.
