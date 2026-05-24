# Selected reference pool report

## Config

- source: `02_dnn/03_reference_search/raw_outputs/36_beta_cell_10_dataset_10_reference/selected_references/`
- selection policy: pool1, L2 top-10, lambda=1, gamma=1 final reference pool

## Output files

- `final_pool1_l2_top10_refs.json`: downstream shell sampling이 읽는 selected reference index이다.
- `policy_comparison_summary.csv`: reference selection policy 비교 summary이다.
- `lambda_sensitivity_summary.csv`: lambda sensitivity summary이다.
- `cell_beta_.../`: selected reference payload copy이며 theta, theta_init, train_summary를 포함한다.

## Reproduction chain

03_reference_search의 `raw_attempts/`와 `selected_references/`에서 pool1/L2/top10 정책으로 selected reference pool을 구성하고, 04_sampling의 모든 radius range가 이 pool을 공통 입력으로 사용한다.
