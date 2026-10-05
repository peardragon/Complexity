# r=1 accuracy 입력

원본: miscellaneous/additional_experiemnts/09_mnist_r1_weighted_accuracy_c100.
Particle 정확도 계산은 replay_one_root.py, 집계는 aggregate_and_plot.py.
raw_outputs/label_noise의 50개 작업 / 500개 reference 관측값에서 CSV 추출.
보존 JSON의 평균·SEM과 차이 0.0 확인.

CSV에는 split 정확도·logZ mixture·reference별 정확도만 있음.
최종 particle 전체나 MNIST 원본은 아님.
순서: reference 10개 평균 → dataset 10개 평균·sample SEM.
원래 statistics.fmean/stdev 사용. Reference 100개를 독립 표본으로 세지 않음.
Study 09의 이전 digit-pair 결과는 이 입력에 포함하지 않음.

재생성: src/make_r1_accuracy.py --execute.
기존 JSON은 건너뜀. --force로 명시적 재생성.
원래 replay JSON에서 가져오려면 --source-dir <dir> --execute --output-dir <dir>.
이 명령은 GPU·SMC를 실행하지 않음.
