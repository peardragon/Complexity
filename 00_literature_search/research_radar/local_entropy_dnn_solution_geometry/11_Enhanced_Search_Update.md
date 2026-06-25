---
title: "Enhanced Search Update"
tags: ["enhanced-search", "miss-prevention", "solution-space", "sampling"]
aliases: ["enhanced literature sweep", "miss-prevention update"]
created: 2026-06-22
source: "enhanced-literature-radar skill + arXiv/OpenAlex/lab-radar outputs"
confidence: high
---

# 11 Enhanced Search Update

## Verdict

Yes. `arXiv:2603.15367` was exactly the kind of recent adjacent prior that the first curated pass could miss: it was already visible inside `lab_radar/riccardo_zecchina/works.jsonl`, but it was not forced through the final manuscript-citation promotion layer until the user supplied it explicitly. The updated sweep now treats user-supplied references as mandatory seeds and also consumes lab-radar latest works, OpenAlex author latest works, arXiv recency queries, and citation-chain candidates.

## Sweep Outputs

- enhanced skill: `/home/bjyong/.codex/skills/enhanced-literature-radar/`
- candidates: `megasearch/enhanced_sweep_latest/candidates.json`
- promoted list: `megasearch/enhanced_sweep_latest/promoted_candidates.md`
- miss-prevention report: `megasearch/enhanced_sweep_latest/miss_prevention_report.md`
- manifest: `megasearch/enhanced_sweep_latest/manifest.json`
- total candidates: 573
- promoted candidates: 133
- user seed check: `2603.15367` found and promoted

## 왜 놓칠 수 있었나

| failure point | observed issue | fix |
| --- | --- | --- |
| user seed gate 부재 | 사용자가 던진 최신 arXiv를 mandatory citation seed로 취급하지 않았다 | `--extra-arxiv 2603.15367`을 강제 fetch하고 found/promoted를 검증 |
| lab latest gate 미흡 | Zecchina/Tiana line의 최신 OpenAlex/lab-radar work가 raw에는 있었지만 curated bibliography에 승격되지 않았다 | `lab_radar/*/works.jsonl`을 직접 parse하여 최신 후보에 가중치 부여 |
| query-only 의존 | keyword sweep은 제목/초록 표현이 조금만 달라도 close prior를 놓칠 수 있다 | core author latest, arXiv author query, citation-chain gate를 병렬 보강 |
| duplicate metadata | OpenAlex DOI 누락 record와 arXiv record가 따로 잡혔다 | title-normalized merge를 추가하여 promoted list를 정리 |

## Curated Newly Surfaced Priors

| bibkey | why it matters | manuscript role | evidence basis |
| --- | --- | --- | --- |
| [@Demyanenko2025GenerativeDiffusion] | perceptron solution space를 generative diffusion으로 sampling하는 최신 통계물리 line | adjacent method, not DNN empirical shell measurement | arXiv + OpenAlex/lab-radar |
| [@Annesi2023StarShapedSpace] | spherical negative perceptron의 solution manifold connectivity를 직접 분석 | theoretical genealogy for connected solution regions | DOI + OpenAlex/lab-radar |
| [@Malatesta2023HighDimensionalManifold] | neural-network solution manifold와 local landscape characterization을 review-style로 묶음 | conceptual bridge from perceptron theory to DNN geometry | arXiv + OpenAlex |
| [@Piccioli2024GibbsSampling] | neural-network posterior를 Gibbs sampler로 직접 sampling하는 최신 sampling prior | adjacent sampling method, posterior rather than reference-local shell | DOI + lab-radar |
| [@Ghiringhelli2026IntermediateTemperatures] | transformer/protein LLM parameter space를 temperature/Langevin 관점에서 분석 | broadening of Zambon/Tiana temperature-sampling line | arXiv |
| [@Ghio2024SamplingFlows] | flow/diffusion/autoregressive sampler를 spin-glass sampling 관점에서 비교 | adjacent generative-sampler context | DOI + OpenAlex/lab-radar |

## Novelty Adjustment

이제 novelty defense는 "solution space를 sampling한 최초 연구"로 쓰면 취약하다. Zambon et al.의 feed-forward NN Boltzmann/Langevin sampling [@Zambon2025SamplingSpace; @Zambon2026ControlledLangevin], posterior Gibbs sampling [@Piccioli2024GibbsSampling], 그리고 perceptron/diffusion sampling [@Demyanenko2025GenerativeDiffusion]이 모두 이미 이 주변을 차지한다.

방어 가능한 축은 더 좁고 선명하다. 본 연구는 global Boltzmann/posterior sampler가 아니라, trained reference pool 주변의 radius shell에서 `phi(d)`와 QC diagnostics를 비교하는 reference-local measurement protocol이다. 또한 novelty는 sampling algorithm 자체가 아니라, theory calibration, MNIST rule-family complexity axis, and QC-aware shell support profiling의 결합에 둬야 한다.

## 본문 삽입 제안

| 넣을 위치 | 본문에 넣을 내용 | 근거 | 주의 |
| --- | --- | --- | --- |
| Related Work | 최근에는 neural-network parameter space를 직접 sampling하려는 흐름이 강화되고 있다. Zambon 등은 feed-forward network의 Boltzmann solution-space sampling을 hybrid Monte Carlo 및 minibatch pseudo-Langevin dynamics로 다루었고, Piccioli 등은 neural-network posterior에 대한 Gibbs sampler를 제안했다 [@Zambon2025SamplingSpace; @Zambon2026ControlledLangevin; @Piccioli2024GibbsSampling]. 따라서 본 연구는 "parameter-space sampling" 자체가 아니라 reference-local shell profile의 측정 문제로 위치를 좁힌다. | Zambon 2025/2026, Piccioli 2024 | "최초 sampling" 주장 금지 |
| Related Work | perceptron 계열에서는 solution manifold의 connected geometry와 효율적 sampling 가능성이 계속 분석되고 있다. spherical negative perceptron의 star-shaped solution space와 generative diffusion 기반 perceptron sampling은, local entropy가 단순 flatness 개념을 넘어 solution-space topology와 sampler design 문제로 확장되고 있음을 보여준다 [@Annesi2023StarShapedSpace; @Demyanenko2025GenerativeDiffusion; @Malatesta2023HighDimensionalManifold]. | PRL + arXiv/OpenAlex | perceptron 결과를 DNN/MNIST에 직접 일반화하지 않는다 |
| Method | 본 연구의 shell estimator는 posterior sampler나 global Boltzmann sampler와 달리, 이미 학습된 reference solution으로부터의 반경별 support를 비교한다. 이 구분은 Gibbs posterior sampling [@Piccioli2024GibbsSampling] 및 minibatch Langevin solution-space sampling [@Zambon2026ControlledLangevin]과의 가장 중요한 방법론적 차이다. | Piccioli 2024, Zambon 2026 | 성능/효율 직접 비교는 하지 않는다 |
| Discussion | intermediate-temperature sampling이 feed-forward networks와 protein transformer models에서 모두 논의되고 있다는 점은, temperature와 solution-space geometry가 최근 독립된 연구축으로 부상했음을 시사한다 [@Zambon2026ControlledLangevin; @Ghiringhelli2026IntermediateTemperatures]. 본 연구의 결과는 이 흐름과 대화할 수 있지만, temperature-optimal training claim이 아니라 fixed-reference local support diagnostic으로 제한된다. | arXiv 2603.15367, 2603.29529 | protein LLM 결과를 MNIST로 과도하게 전이하지 않는다 |
| Limitation | 강화된 문헌 탐색 이후 본 연구의 가장 약한 novelty framing은 "신경망 해공간을 sampling한다"는 넓은 표현이다. global sampler, posterior sampler, diffusion sampler prior art가 이미 존재하므로 [@Demyanenko2025GenerativeDiffusion; @Piccioli2024GibbsSampling; @Zambon2026ControlledLangevin], 본문에서는 `reference-local`, `QC-aware`, `rule-complexity-conditioned`라는 세 수식어를 반복적으로 유지해야 한다. | enhanced sweep promoted candidates | contribution 범위를 좁혀야 한다 |

## 반드시 읽어야 할 추가 PDF

1. [@Zambon2026ControlledLangevin] Controlled Langevin Dynamics for Sampling of Feedforward Neural Networks Trained with Minibatches - https://arxiv.org/pdf/2603.15367
2. [@Zambon2025SamplingSpace] Sampling the space of solutions of an artificial neural network - https://arxiv.org/pdf/2503.08266
3. [@Demyanenko2025GenerativeDiffusion] Generative diffusion for perceptron problems - https://arxiv.org/pdf/2502.16292
4. [@Malatesta2023HighDimensionalManifold] High-dimensional manifold of solutions in neural networks - https://arxiv.org/pdf/2309.09240
5. [@Ghiringhelli2026IntermediateTemperatures] Sampling at intermediate temperatures is optimal for training large language models in protein structure prediction - https://arxiv.org/pdf/2603.29529

## Backlinks

- [[00_Index]]
- [[02_Prior_Work_Map]]
- [[05_Claim_Evidence_Matrix]]
- [[06_Strongest_Prior_Art]]
- [[10_2026_Sampling_Addendum]]
