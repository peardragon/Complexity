# r=1 accuracy 입력

원본: miscellaneous/additional_experiemnts/23_digit_pair_cms_10dataset_mean_rank_production/downstream.
Particle 정확도 계산은 src/replay_r1_accuracy.py, 집계는 src/aggregate_r1_and_seal.py.
r1_accuracy/raw_outputs/digit_pair의 120개 작업 / 1,200개 reference에서 CSV 추출.
보존 JSON의 평균·SEM과 차이 0.0 확인.

90개 replay와, 동일 학습 입력·reference의 이전 관측을 재사용한 30개 작업임.
30개 작업은 pair_4_9, pair_3_8, pair_4_6. 현재 r=1 logZ·직접 미분도 일치.
pair_4_6의 이전 순위 metadata는 37, 현재는 33. 학습 배열은 같음.
CSV는 이전 순위·metadata hash를 복제하지 않고 condition/dataset/reference로 식별.

CSV에는 split 정확도·logZ mixture·reference별 정확도만 있음.
최종 particle 전체나 MNIST 원본은 아님.
순서: reference 10개 평균 → dataset 10개 평균·sample SEM.
원래 statistics.fmean/stdev 사용. Study 09의 이전 12쌍을 사용하지 않음.

재생성: src/make_r1_accuracy.py --execute.
기존 JSON은 건너뜀. --force로 명시적 재생성.
원래 replay JSON에서 가져오려면 --source-dir <dir> --execute --output-dir <dir>.
이 명령은 GPU·SMC를 실행하지 않음.
