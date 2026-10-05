# r=1 accuracy 입력

원본: miscellaneous/additional_experiemnts/08_synthetic_r1_weighted_accuracy_c100.
Particle 정확도 계산은 replay_r1_accuracy.py, 집계는 aggregate_and_plot.py.
그때 저장한 1,080개 작업 / 10,800개 reference 관측값에서 CSV를 추출함.
보존 JSON의 평균·SEM과 차이 0.0 확인.

CSV에는 split 정확도·logZ mixture·reference별 정확도만 있음.
최종 particle 전체나 training dataset은 아님.
순서: reference 10개 평균 → dataset 60개 평균·sample SEM.
원래 NumPy mean/std(ddof=1) 사용.

재생성: src/make_r1_accuracy.py --execute.
기존 JSON은 건너뜀. --force로 명시적 재생성.
원래 replay JSON에서 가져오려면 --source-dir <dir> --execute --output-dir <dir>.
이 명령은 GPU·SMC를 실행하지 않음.
