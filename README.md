# Complexity

논문: [Dataset Complexity Shapes Finite-Distance Loss Geometry in Neural Networks](https://arxiv.org/abs/2608.22361v1).

| 폴더 | 논문 내용 |
|---|---|
| 01_theory | Perceptron RS·finite-size SMC, Fig. 2, Appendix B, Supplement |
| 02_dnn_synthetic | Synthetic DNN, Figs. 3–4 |
| 03_dnn_mnist | Label noise·digit pairs, Figs. 5–8 |
| 04_discussion | Normalized radial response·대칭 방향·재진입, Fig. 9, Appendix D |
| Figures | 본문 Fig. 1–9와 Discussion 보조 그림 |

Linux/Python 3.12에서 검증. 확인한 package 버전은 requirements에 고정.
설치는 `python -m pip install -r requirements.txt`.
Notebook의 원래 글꼴로 다시 그리려면 LaTeX와 Latin Modern도 필요함.
GPU sampling에는 CUDA 지원 PyTorch와 사용할 GPU 지정이 필요함.

```bash
python Figures/paper_figures/src/validate_release.py
python Figures/paper_figures/src/build_all.py
python Figures/04_discussion/src/make_figures.py
```

기존 그림은 건너뜀. `build_all.py --force`는 notebook의 현재 설정으로 다시 그림.
Notebook은 `Figures/paper_figures/releases/rebuild_all_paper_figures.ipynb`.
실행 출력과 그림을 포함해서 보관.

Raw → summary → figure 흐름:

| 실험 | Raw 입력 | Summary | Figure |
|---|---|---|---|
| Theory | 고정 dataset/reference pool, shell shard | phi_by_sampling.csv | Fig. 2 |
| Synthetic | dataset.npz → reference pack → odd/even shell shard | cms_by_dataset.csv, energetic_profiles.csv, condition_metrics.csv | Figs. 3–4 |
| MNIST | dataset.npz → reference pack → 조건별 shell shard | C_MS, energetic profile, condition metrics | Figs. 5–8 |
| Discussion | 02/03 profile, clean/random antipodal NPZ | h=-g_E/r, K_r, sign intervals | Fig. 9, Appendix D |

큰 raw dataset·reference·particle shard는 배포에서 제외. 작은 그림 입력 NPZ 4개와
요약 CSV·Appendix D compact 입력은 포함하므로 raw 없이 그림을 확인할 수 있음.
Raw가 있으면 각 요약 스크립트의 `--execute --output-dir <dir>`로 별도 재생성 가능.
Theory는 `--aggregate --output-dir <dir>` 사용.

이전 scalar raw에 r=1 terminal accuracy가 없으면 보존된 accuracy JSON을 사용하며
실행 로그에 archival source를 표시함. 새 sampling은 weighted training accuracy를 저장.
Theory의 고정 input pool 생성은 별도 준비가 필요하며, 이 repo의 확인 명령은 기존 pool을 검사함.

Figure 9(b,c)는 개념도. 측정된 loss slice나 Hessian spectrum이 아님.
보존된 old raw의 legacy full-shell 보조값은 재집계에 사용하지 않음.
Summary는 논문의 squared-distance delta `(P-2)/P` 관례로 재구성.
