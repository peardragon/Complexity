# Complexity Reorganization Production Audit

작성 시각: 2026-06-30 KST

## 최종 기준 root

- 최종 산출 tree: `/home/bjyong/Complexity/Complexity`
- Git 반영 대상 repo: `/home/bjyong/Complexity/local_project`
- raw 백업/이전본 참조 허용 범위: `/home/bjyong/Complexity/reorg_backups`

## 4 rules / eta sweep direct derivative 교체

MNIST proxy-local-entropy 최종본은 sampling-time direct radial score derivative를 사용한다.

- eta sweep 최종 raw: `03_dnn_mnist/label_noise_sweep/04_sampling/raw_outputs/shell_pool/noise_eta_*/ref_*/r_*/samples.npz`
- eta sweep 최종 unit summary: `03_dnn_mnist/label_noise_sweep/04_sampling/summarized_outputs/direct_derivative_methodology_30ref_r0p01_to_1p0_step0p01_n1024_cpu60_gpu0/02_eta_flip_sampling/05_pool2_pm_sais_sampling/shell_summary_by_unit_with_phi_derivatives.csv`
- eta sweep 최종 축: `eta=0.05, 0.15, 0.25`
- eta sweep 최종 unit count: `9000`
- manual 4 rules 최종 raw: `03_dnn_mnist/manual_rules/04_sampling/raw_outputs/shell_pool/rule_*/ref_*/r_*/samples.npz`
- manual 4 rules 최종 unit summary: `03_dnn_mnist/manual_rules/04_sampling/summarized_outputs/direct_derivative_methodology_30ref_r0p01_to_1p0_step0p01_n1024_cpu60_gpu0/01_active_rules_sampling/05_pool2_pm_sais_sampling/shell_summary_by_unit_with_phi_derivatives.csv`
- manual 4 rules 최종 unit count: `12000`

최종 derivative 컬럼은 다음을 anchor로 한다.

- unit-level: `dlogZ_inf_full_dr`, `d_phi_energy_direct_dd_unit`, `split_dlogZ_dr_per_P_diff`
- eta group: `d_phi_energy_direct_dd`, `d_phi_energy_direct_dd_sem`
- manual group: `d_phi_energy_direct_dd_unit_mean`, `d_phi_energy_direct_dd_unit_sem`

기존 finite-difference 기반 proxy payload와 eta=0.02 PLE payload는 최종 PLE 정책에서 제외했다. 단, `01_dataset`/`02_complexity_measure` stage의 eta=0.02 dataset/complexity summary는 해당 stage 자체의 자료로 남아 있다.

## Sampling 정책 확인

최종 정책은 importance sampling with tempered path이다.

- MNIST eta/manual raw unit summary는 `sampler_method=exact_shell_l2_vmf_adaptive_ce_tempered_smc`로 통일되어 있다.
- MNIST eta/manual derivative methodology는 `mnist10_exact_shell_l2_vmf_ce_tempered_smc_radial_score_derivative_v1`이다.
- theory sampling은 `choose_next_temperature`, target CESS, resampling ESS로 구성된 CE-tempered SMC path를 사용한다. summary의 legacy method label은 `exact_shell_l2_vmf_adaptive_ce_smc`이지만, run id와 validation은 default SMC tempered path replacement를 명시한다.
- 3NN synthetic/random baseline은 adaptive CE SMC에서 temperature path를 target CESS로 선택한다. 기존 raw metadata 일부는 legacy label `exact_shell_l2_vmf_adaptive_ce_smc`를 보존하지만, source/config는 새 production label `exact_shell_l2_vmf_adaptive_ce_tempered_smc`로 정리했다.

3NN synthetic sampling source 독립성을 위해 다음 stage-local helper를 추가했다.

- `02_dnn_synthetic/04_sampling/src/io_utils.py`
- `02_dnn_synthetic/04_sampling/src/loaders.py`

## QC 결과

QC 정책은 두 개만 최종 그림/summary로 유지한다.

- A. reference/dataset 평균 대표성이 타당한가: reference variability, dataset variability where applicable
- B. unit-level logZ 계산이 split 기준으로 안정적인가: split logZ per parameter

ESS는 tempered path 내부 control로 보고, 별도 figure policy에서는 제외한다.

| stage | QC rows | claim |
|---|---:|---|
| theory sampling logZ stability | 168 | pass 168 |
| 3NN synthetic logZ split QC | 162 | pass 162 |
| 3NN synthetic random baseline logZ split QC | 9 | pass 9 |
| MNIST eta logZ split QC | 21 | pass 9, inspect 12 |
| MNIST manual rules logZ split QC | 400 | pass 94, inspect 306 |

MNIST의 `inspect`는 strict threshold 기준의 표시이다. 최종 raw/summary는 tempered path와 direct derivative를 사용하지만, split logZ QC threshold 기준에서는 추가 검토 대상으로 남긴다.

## Figures

Top-level figure bundle:

- root: `Figures/`
- PNG count: `47`
- rebuild script: `Figures/rebuild_figures.py`
- notebook: `Figures/rebuild_all_figures_from_summaries.ipynb`

Notebook 정책:

- 각 top-level figure당 code cell 하나를 둔다.
- `src` visualization helper를 import하지 않는다.
- `summarized_outputs` 또는 figure-input CSV만 읽는다.
- 기본 output은 `Figures/notebook_regenerated/`이다.

random baseline dataset example은 raw 없이 notebook에서 재생성 가능하도록 summary point CSV를 추가했다.

- `02_dnn_synthetic/06_random_baseline/01_dataset/summarized_outputs/gaussian_random_90_dataset/example_dataset_points.csv`

## Standalone 재생성 명령

최상위 figure bundle:

```bash
cd /home/bjyong/Complexity
python Complexity/Figures/rebuild_figures.py
```

MNIST eta sampling path/index validation:

```bash
cd /home/bjyong/Complexity
python Complexity/03_dnn_mnist/label_noise_sweep/04_sampling/src/stage_paths.py
```

MNIST manual unit index:

```bash
cd /home/bjyong/Complexity
python Complexity/03_dnn_mnist/manual_rules/04_sampling/src/build_unit_index.py
```

MNIST proxy figures:

```bash
cd /home/bjyong/Complexity
python Complexity/03_dnn_mnist/label_noise_sweep/05_proxy_local_entropy/src/build_six_figures.py
python Complexity/03_dnn_mnist/label_noise_sweep/05_proxy_local_entropy/src/build_qc_figures.py
python Complexity/03_dnn_mnist/manual_rules/05_proxy_local_entropy/src/build_summary_inputs.py
python Complexity/03_dnn_mnist/manual_rules/05_proxy_local_entropy/src/build_six_figures.py
python Complexity/03_dnn_mnist/manual_rules/05_proxy_local_entropy/src/build_qc_figures.py
```

## 검증 완료

- `python Complexity/Figures/rebuild_figures.py`: 성공, 47개 top-level PNG 재생성
- `Figures/rebuild_all_figures_from_summaries.ipynb`: setup + 47 figure cells 실행 성공
- `python Complexity/03_dnn_mnist/label_noise_sweep/04_sampling/src/stage_paths.py`: 성공
- `python Complexity/03_dnn_mnist/manual_rules/04_sampling/src/build_unit_index.py`: 성공
- `python -m py_compile` for patched source files: 성공
- `python -m json.tool` for patched config: 성공
