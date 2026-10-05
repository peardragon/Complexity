# 01_theory

Perceptron RS 해와 finite-size SMC. Fig. 2, Appendix B, Supplement S1–S8.

- `01_theory_analytic`: alpha=0.1의 RS 해.
- `02_theory_sampling`: N=40,80,160,320. Dataset 10개 × reference 10개.
  기본 particle 수 2^15, N=320에서 particle 수 비교.
- 설정은 각 `config/default.json`. 실행은 `src`, 보조 함수는 `src/utils`.

```bash
python 01_theory/01_theory_analytic/src/theory_full_rs.py
python 01_theory/02_theory_sampling/src/make_summarized_outputs.py
```

기존 결과는 건너뜀. RS 재계산은 `--force`.
`make_datasets.py`, `make_references.py`는 기존 input pool 확인용.
Raw 재계산에는 해당 pool이 필요함.

```bash
python 01_theory/02_theory_sampling/src/make_summarized_outputs.py --check-only
python 01_theory/02_theory_sampling/src/make_summarized_outputs.py --aggregate --output-dir /path/to/comparison
```

두 번째 명령은 기존 raw shard가 있을 때 사용.
평균 순서는 reference → dataset. Full-shell 항은 논문의 squared-distance
delta 관례 `(N-2)/N`으로 재구성하며, old raw의 legacy full-shell 보조값은 사용하지 않음.
Input pool·shard는 배포 제외. RS와 SMC 요약 CSV는 포함.
