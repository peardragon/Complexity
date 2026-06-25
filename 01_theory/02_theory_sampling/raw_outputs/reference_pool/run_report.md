# Two-pool sampling reference_pool run report

이 폴더는 reference pool sampling 결과와 reference bias/reweighting summary를 보관한다.

## Output files

- `reference_pool_summary.csv`: reference pool 구성과 reference-level summary를 담는다.
- `reference_sampling_bias_reweighting.csv`: reference sampling bias 보정 및 reweighting 값을 담는다.

## Downstream use

`shell_pool/sample_unit_summary.csv` 및 comparison stage에서 reference normalization과 QC 설명을 재현할 때 사용된다.
