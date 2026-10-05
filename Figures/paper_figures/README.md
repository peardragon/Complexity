# Paper figures

arXiv:2608.22361v1, Fig. 1–9.

| 번호 | 내용 |
|---|---|
| 1 | Dataset-to-landscape 개념도 |
| 2 | Perceptron benchmark |
| 3–4 | Synthetic |
| 5–6 | MNIST label noise |
| 7–8 | MNIST digit pairs |
| 9 | Normalized radial response와 corridor 개념도 |

```bash
python Figures/paper_figures/src/build_all.py
python Figures/paper_figures/src/build_all.py --check-only
```

기존 출력은 파일별로 건너뜀. 누락 형식만 생성.
`build_all.py --force`는 notebook의 현재 설정으로 전부 다시 그림.
원본 요약이 바뀌면 staged 입력과 해당 그림을 함께 갱신.
크기·간격·범례는 `releases/rebuild_all_paper_figures.ipynb`의 `conf`에서 수정.
Notebook은 실행 출력과 그림을 포함해 그대로 배포.

- `config/figure_manifest.json`: 번호, renderer, 입력, 출력 경로.
- `config/input_sources.json`: 원본 → 그림 입력 대응. 기본 build에서 동기화.
- `../config/paper_figure_style.json`: 공통 스타일.
- `src/utils`: 스타일 및 notebook 보조 함수.

Fig. 9는 04 Discussion에서 동기화한 요약을 읽음.
전체 확인은 `src/validate_release.py`. Raw 없이도 검증·그림 생성 가능.
원고 수정·원격 배포는 하지 않음.
