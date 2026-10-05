# Digit-pair experiment

45쌍 각각 dataset 10개의 평균 C_MS로 정렬. 1,5,9,…,45위의 12쌍 사용.
Figs. 7–8에 해당.

```text
4/9, 3/8, 5/9, 2/7, 4/5, 0/2,
2/9, 7/8, 4/6, 1/6, 0/9, 0/1
```

목록은 각 단계의 `config/frozen_pair_manifest.json`.
C_MS와 QC는 120행, profile은 1,200행, condition summary는 12행.
일부 seed가 pair 간 재사용됨. 실행 방법은 상위 README 참고.

01은 각 replica의 45쌍 C_MS를 해당 raw `pair_manifest.json`에만 기록한다.
02가 열 replica를 평균하고 결정적 tie-break `(digit_a, digit_b)`를 적용해
`summarized_outputs/digit_pairwise_complexity_summary.json`을 소유·생성한다.
이 JSON의 선택 rank와 평균은 frozen manifest와 수치적으로 대조된다.
