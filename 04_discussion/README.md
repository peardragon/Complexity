# 04_discussion

기준: [arXiv:2608.22361v1](https://arxiv.org/html/2608.22361v1).

| 폴더 | 계산 | 논문 대응 |
|---|---|---|
| 01_frozen_inputs | 입력 경로·행 수 정리 | 실행 준비용 |
| 02_normalized_radial_geometry | h(r)=-g_E(r)/r, peak와 turning scale | Discussion: Finite-distance hardening, Fig. 9(a) |
| 03_antipodal_geometry | Clean/random의 ±방향 대칭 응답 K_r | MNIST label-noise Results 마지막 문단, Appendix D |
| 04_random_label_reentrance | g_E의 음–양–음 구간, zero crossing | Fig. 6(b), Discussion: Random limits and natural data |

Fig. 9(b,c)는 constraint-crowding 개념도. 그림 코드는 Figures의 04_discussion에 있음.
고정 Hessian으로 h의 상승을 설명할 수 없다는 부분은 본문의 수식 유도에 해당.
이 폴더에서 Hessian spectrum을 계산하는 것은 아님.

```bash
python 04_discussion/src/run_all.py
python 04_discussion/src/validate_all.py
```

기존 요약은 출력 파일별로 건너뜀. 재계산은 `run_all.py --force`.
설정은 각 `config/default.json`, 보조 함수는 `src/utils`.
검증 명령은 파일을 수정하거나 receipt를 작성하지 않는다. 코드·환경 hash는
실행 및 파일 재사용을 막는 조건으로 사용하지 않는다.

다른 위치에서 전체 흐름을 비교하려면 다음처럼 실행한다. 이 경우 네 단계의
`summarized_outputs`가 지정한 디렉터리 아래에 같은 구조로 생성된다.

```bash
python 04_discussion/src/run_all.py --output-dir /path/to/comparison
python 04_discussion/src/validate_all.py --output-dir /path/to/comparison
```

Appendix D의 원래 평가 NPZ는 `03_antipodal_geometry/raw_outputs`에 있으며
현재 release policy에 따라 Git 배포에서는 제외한다. 두 NPZ가 모두 있으면 Stage 03은 원래
dataset-block 통계, t 구간, exact sign-flip, 방향 비율, clean/random paired
contrast와 peak contrast를 다시 계산해 네 `matched_antipodal_*` 표를 만든다.
두 NPZ가 모두 없으면 `frozen_inputs`의 네 compact 공개 표에서 같은 출력
schema를 만든다. 하나만 있는 불완전한 raw pair는 오류다.
평가 조건: dataset 10개 × reference 10개 × directions 256개,
r={.01,.03,.07,.10,.12,.20,.40}, seed=2026082201.

```bash
python 04_discussion/03_antipodal_geometry/src/evaluate_antipodal.py \
  --condition noise_eta_0p50 --device cuda:0 --execute
python 04_discussion/03_antipodal_geometry/src/make_summarized_outputs.py --force
```

`--device`는 평가 장치만 바꾸는 실행 보조 설정이다. 과학 설정인 objective
scale, parameter 수, radius, seed/stride, direction 수와 chunk 크기는 Stage 03의
`config/default.json`에 있다. 평가 명령은 학습이나 SMC를 하지 않으며 raw NPZ만
만든다. 요약 명령은 NPZ가 있으면 raw에서 네 공개 schema를 재구성한다.
`--output-dir`과 `--force`는 평가 및 요약 명령에도 제공된다. 평가 NPZ를
비교 디렉터리에 쓴 뒤 그 파일을 요약할 때는 요약 명령에
`--raw-dir /path/to/comparison/03_antipodal_geometry/raw_outputs`를 함께 준다.
별도 원본 대조는 `src/validate_release_scope.py`.
