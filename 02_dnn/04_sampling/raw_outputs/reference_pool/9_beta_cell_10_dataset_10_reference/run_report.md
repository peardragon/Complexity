# Sampling reference_pool report

이 reference_pool은 theory stage와 parallel한 구조를 유지한다. active data payload는 `selected_reference_pool/` 하나이며, distance-range별 reference_pool wrapper는 report-only였으므로 백업 이동했다.

## Downstream use

`02_dnn/04_sampling/config/*.yaml`의 `reference_search_root`가 이 selected reference pool을 입력으로 사용한다.
