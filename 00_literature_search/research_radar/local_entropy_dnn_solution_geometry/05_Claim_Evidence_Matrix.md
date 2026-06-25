---
title: "Claim Evidence Matrix Expanded"
tags: ["claims", "evidence", "expanded", "recent-citation-chase"]
aliases: ["claim evidence matrix expanded"]
created: 2026-06-22
source: "curated literature + OpenAlex citation chase + arXiv/DOI verification + local project reports"
confidence: high
---

# 05 Claim Evidence Matrix Expanded

## 확장 요약

- verified reference coverage: **60 refs** (`07_Bibliography.bib` 기준)
- 새로 추가한 검증 레퍼런스: **25 refs**
- recent citing check: top prior art 6개 축에 대해 OpenAlex recent-citer 후보를 수집하고, DOI/arXiv/공식 venue가 확인된 항목만 matrix에 반영했다.
- 가장 중요한 방어 문장: 본 연구는 **optimizer**, **universal flatness/generalization predictor**, **global density-of-states**, **global mode-connectivity proof**가 아니라 **reference-local, QC-aware, rule-complexity-conditioned shell support profile**을 측정하는 protocol이다.

## Claim Evidence Matrix

| claim | evidence | refs | confidence | insertion_point | basis |
| --- | --- | --- | --- | --- | --- |
| local entropy는 reference 주변 solution density/free energy를 측정하는 quantity다 | Franz-Parisi potential, CSP local entropy, DNN partial local entropy가 모두 fixed reference/local neighborhood 관점을 공유한다. | [@Franz1995RecipesMetastable; @Baldassi2016LocalEntropy; @Musso2021PartialLocalEntropy] | high | Introduction | PDF/abstract+metadata |
| dense/wide solution regions는 rare하지만 algorithmically accessible할 수 있다 | discrete synapse/perceptron 연구와 최근 binary perceptron theorem이 rare well-connected clusters를 보강한다. | [@Baldassi2015SubdominantDense; @Baldassi2016UnreasonableEffectiveness; @Abbe2022BinaryPerceptron] | high | Related Work | PDF/abstract+OpenAlex citing check |
| 최근 perceptron theory는 local entropy 계보를 강화하면서도 typical-solution caveat를 만든다 | frozen 1-RSB, atypical connected states, overlap gap results는 dense cluster와 typical geometry가 다를 수 있음을 보여준다. | [@Perkins2021FrozenRSB; @Baldassi2023TypicalAtypical; @Barbier2025EscapeAtypical; @Benedetti2025OverlapGap] | high | Related Work/Limitation | DOI/arXiv+recent web verification |
| 본 연구의 novelty는 optimizer가 아니라 reference-local measurement protocol이다 | Entropy-SGD/SAM/ASAM/LookSAM/GNAM/CR-SAM은 neighborhood 정보를 optimization objective에 넣는 계열이다. | [@Chaudhari2017EntropySGD; @Foret2021SharpnessAware; @Kwon2021ASAM; @Liu2022EfficientSAM; @Zhang2023GradientNormAware; @Wu2024CRSAM] | high | Method/Related Work | PDF/metadata+recent citing chase |
| flatness-generalization 인과를 직접 주장하면 prior art에 취약하다 | sharpness는 rescaling, optimizer, architecture, metric choice에 따라 correlation이 깨질 수 있다. | [@Dinh2017SharpMinima; @Zhang2021WhyFlatness; @Jiang2020FantasticGeneralization; @Andriushchenko2023ModernSharpness; @Abdollahpoorrostam2024CLIPSharpness] | high | Limitation/Discussion | PDF/arXiv+OpenAlex citing check |
| scale/permutation symmetry는 raw parameter-shell 해석을 제한한다 | ASAM과 mode-connectivity/symmetry papers는 parameter-space metric이 quotient geometry와 다를 수 있음을 보여준다. | [@Kwon2021ASAM; @Pittorino2022DeepNetworksToroids; @Entezari2022PermutationModeConnectivity; @Ainsworth2023GitReBasin] | high | Limitation | PDF/arXiv |
| local smoothness/support와 global connectivity는 구분해야 한다 | loss landscape taxonomy와 mode connectivity 문헌은 local metric, global connectedness, ensemble similarity를 별도 축으로 본다. | [@Yang2021TaxonomizingLandscape; @Garipov2018LossSurfaces; @Draxler2018EssentiallyNoBarriers; @Li2018VisualizingLoss] | high | Discussion | PDF/arXiv |
| dataset/rule complexity axis는 random labels만이 아니라 structured-data theory와 연결된다 | random-label memorization, double descent, hidden manifold/data-structure papers가 label/data complexity controls를 정당화한다. | [@Zhang2017RethinkingGeneralization; @Arpit2017CloserMemorization; @Nakkiran2020DeepDoubleDescent; @Belkin2019ReconcilingBiasVariance; @Goldt2020HiddenManifold] | high | Experiment | PDF/arXiv+local reports |
| graph total variation/NMSTV는 label smoothness를 표현하는 보조 complexity axis로 쓸 수 있다 | graph signal processing과 local-global consistency는 graph 위 smooth signal 해석의 근거를 제공한다. | [@Zhou2004LearningLocalGlobal; @Shuman2013EmergingGraphSignal; @Ortega2018GraphSignalProcessing] | medium | Experiment | metadata+local NMSTV report |
| theory arm은 DNN claim을 바로 증명하는 것이 아니라 estimator calibration layer다 | perceptron statistical mechanics와 recent perceptron geometry가 shell estimator의 toy-theory validation 위치를 만든다. | [@Gardner1988SpaceInteractions; @Seung1992StatisticalMechanics; @Franz1995RecipesMetastable; @Abbe2021ProofContiguity] | high | Method | PDF/metadata+local theory reports |
| shell partition/logZ estimation은 AIS/SMC/directional sampling foundation 위에 있다 | normalizing constant estimation과 vMF sampling은 estimator 방법론의 직접 근거다. | [@Neal2001AnnealedImportance; @DelMoral2006SMCSamplers; @Wood1994SimulationVMF] | high | Method | PDF/metadata |
| global density-of-states는 가장 가까운 adjacent measurement prior다 | Wang-Landau 계열은 global DOS를 추정하고, 최근 NN-DOS 연구는 dataset structure와 loss spectrum을 연결한다. | [@Wang2001FlatHistogram; @Liu2023GradientWangLandau; @Mele2025DensityStates; @Winer2026DeepNeuralNetsHamiltonians] | high | Related Work | PDF/arXiv/OpenReview |
| SGD dynamics와 flatness preference는 batch size/learning-rate regime에 민감하다 | SGD noise scale과 2024 SGD regime paper는 reference pool 생성 조건을 protocol에 포함해야 함을 시사한다. | [@Smith2018BayesianPerspective; @Keskar2017LargeBatch; @Sclocchi2024DifferentRegimesSGD; @Ly2025MultifractalLandscapes] | medium-high | Method/Discussion | PDF/arXiv/recent DOI |
| PAC-Bayes link는 해석 가능하지만 본 연구가 bound를 계산한 것은 아니다 | PAC-Bayes flatness literature는 formal bridge를 주지만 phi(d)는 empirical diagnostic으로 제한해야 한다. | [@Dziugaite2017ComputingNonvacuous; @Foret2021SharpnessAware; @Haddouche2025PACBayesianLink] | high | Discussion | PDF/PMLR |
| MNIST 현재 결과는 diagnostic evidence와 promotion-ready claim을 분리해야 한다 | 90-ref run은 complete이나 QC pass subset이 제한되어 full-grid exploratory와 QC-passed stronger evidence를 구분해야 한다. | local reports + [@Yang2021TaxonomizingLandscape; @Andriushchenko2023ModernSharpness] | high | Experiment/Limitation | local reports+literature caveat |
| novelty 방어의 핵심 문장은 'reference-local, QC-aware, rule-complexity-conditioned support profile'이다 | optimizer, universal flatness measure, global DOS, global connectivity와 겹치지 않는 기여 축이다. | [@Baldassi2016LocalEntropy; @Mele2025DensityStates; @Yang2021TaxonomizingLandscape; @Wang2001FlatHistogram] | high | Introduction/Discussion | synthesis |

## Recent Citation Chase 요약

| top prior art seed | 최근 확인한 citing/adjacent papers | novelty 판단에 주는 의미 |
| --- | --- | --- |
| [@Baldassi2016LocalEntropy] | [@Musso2021PartialLocalEntropy; @Abbe2022BinaryPerceptron; @Baldassi2023TypicalAtypical; @Catania2024CopycatPerceptron; @Barbier2025EscapeAtypical; @Benedetti2025OverlapGap] | local entropy는 DNN weight anisotropy와 perceptron connected/atypical solution theory로 확장됐다. 본 연구는 이 흐름을 DNN rule-family shell measurement로 옮기는 위치다. |
| [@Chaudhari2017EntropySGD] | [@Kwon2021ASAM; @Liu2022EfficientSAM; @Zhang2023GradientNormAware; @Wu2024CRSAM] | 최근 인용 흐름은 대부분 optimizer 개선이다. 따라서 본 연구가 optimizer 성능 개선이 아니라 measurement protocol임을 강하게 분리해야 한다. |
| [@Dinh2017SharpMinima; @Jiang2020FantasticGeneralization] | [@Zhang2021WhyFlatness; @Kwon2021ASAM; @Yang2021TaxonomizingLandscape; @Andriushchenko2023ModernSharpness; @Abdollahpoorrostam2024CLIPSharpness] | 최근 연구는 flatness metric의 취약성과 조건부 유효성을 동시에 보여준다. phi(d)는 universal generalization predictor가 아니라 fixed protocol diagnostic으로 써야 한다. |
| [@Garipov2018LossSurfaces; @Draxler2018EssentiallyNoBarriers] | [@Entezari2022PermutationModeConnectivity; @Ainsworth2023GitReBasin; @Abdollahpoorrostam2024CLIPSharpness] | mode connectivity는 symmetry quotient와 interpolation 관점으로 발전했다. 사용자의 radial shell profile은 global basin connectivity claim의 대체물이 아니다. |
| [@Zhang2017RethinkingGeneralization] | [@Nakkiran2020DeepDoubleDescent; @Belkin2019ReconcilingBiasVariance; @Goldt2020HiddenManifold; @Mele2025DensityStates] | random-label control은 단순 stress test에서 dataset structure/complexity와 loss-space volume을 연결하는 흐름으로 확장됐다. |
| [@Mele2025DensityStates] | [@Wang2001FlatHistogram; @Liu2023GradientWangLandau; @Winer2026DeepNeuralNetsHamiltonians; @Ly2025MultifractalLandscapes] | Mele는 아직 너무 최근이라 citing corpus가 작다. 대신 Wang-Landau/DOS lineage와 2025 loss-landscape model을 함께 써 adjacent prior로 다룬다. |

## 본문 삽입용 추가 문단

### 추가 삽입 문단 1: Related Work

- 넣을 위치: Related Work
- 본문에 넣을 내용: 최근 local entropy 계열은 단순히 wide minima를 경험적으로 관찰하는 수준을 넘어, binary perceptron에서 rare well-connected cluster와 atypical connected solution의 존재를 이론적으로 분석하는 방향으로 확장되고 있다 [@Abbe2022BinaryPerceptron; @Barbier2025EscapeAtypical]. 본 연구는 이러한 connected-cluster theory를 직접 증명하지는 않지만, trained reference 주변의 shell-wise support profile을 측정함으로써 DNN 설정에서 유사한 문제의식을 경험적으로 추적한다.
- 근거/주의: perceptron 최신 theory를 novelty 배경으로 넣되 직접 등식 주장 금지.

### 추가 삽입 문단 2: Limitation

- 넣을 위치: Limitation
- 본문에 넣을 내용: parameter-space geometry를 해석할 때는 scale invariance와 permutation symmetry가 중요한 교란요인이다 [@Kwon2021ASAM; @Entezari2022PermutationModeConnectivity; @Ainsworth2023GitReBasin]. 따라서 본 연구의 L2-shell 결과는 quotient-space geometry가 아니라, 명시된 architecture와 coordinate convention에서의 reference-local diagnostic으로 보고한다.
- 근거/주의: symmetry/reparameterization caveat 강화.

### 추가 삽입 문단 3: Discussion

- 넣을 위치: Discussion
- 본문에 넣을 내용: local sharpness와 global landscape structure는 같은 정보가 아니다. 대규모 empirical taxonomy는 local smoothness, global connectivity, ensemble similarity가 서로 구분되는 축임을 보였고 [@Yang2021TaxonomizingLandscape], mode-connectivity 연구 역시 minima의 isolated basin 해석을 약화시킨다 [@Garipov2018LossSurfaces; @Draxler2018EssentiallyNoBarriers].
- 근거/주의: global topology claim 방지.

### 추가 삽입 문단 4: Experiment

- 넣을 위치: Experiment
- 본문에 넣을 내용: dataset/rule complexity를 실험 설계에 포함하는 것은 random labels가 memorization control로 기능한다는 고전적 관찰뿐 아니라 [@Zhang2017RethinkingGeneralization; @Arpit2017CloserMemorization], data structure가 학습 dynamics와 generalization에 미치는 영향을 명시적으로 모델링하려는 최근 통계물리적 흐름과도 맞닿아 있다 [@Goldt2020HiddenManifold; @Nakkiran2020DeepDoubleDescent].
- 근거/주의: MNIST rule-family design justification.

### 추가 삽입 문단 5: Related Work

- 넣을 위치: Related Work
- 본문에 넣을 내용: global density-of-states 접근은 전체 parameter 또는 output space의 volume distribution을 추정한다는 점에서 본 연구와 가장 가까운 adjacent prior다 [@Wang2001FlatHistogram; @Liu2023GradientWangLandau; @Mele2025DensityStates]. 그러나 본 연구는 global DOS가 아니라 trained reference로부터의 거리별 shell support를 추정하므로, 질문의 단위가 전체 landscape에서 reference-local neighborhood로 이동한다.
- 근거/주의: Mele/Wang-Landau와의 차별화.

## 주의 문장

- `phi(d)`를 일반화 성능의 보편 예측자로 쓰지 않는다.
- L2-shell 결과를 permutation/symmetry quotient geometry로 해석하지 않는다.
- MNIST 90-ref full grid는 완료되었지만 QC-passed subset과 exploratory full-grid를 분리한다.
- global density-of-states, mode connectivity, optimizer improvement와 직접 경쟁하는 contribution으로 쓰지 않는다.

## 2026 Sampling Addendum

사용자가 지적한 [@Zambon2026ControlledLangevin]은 기존 60-reference core 이후 확인한 매우 가까운 최신 prior art다. 이 논문은 minibatch pseudo-Langevin dynamics로 feed-forward neural-network parameter space를 Boltzmann distribution에 따라 sampling하고, exact hybrid Monte Carlo와 equilibrium statistics를 비교한다. 따라서 density-of-states [@Mele2025DensityStates]와 함께, 본 연구가 반드시 구분해야 할 최신 adjacent prior로 분류한다.

| claim | evidence | refs | confidence | insertion_point | basis |
| --- | --- | --- | --- | --- | --- |
| scalable sampling of NN parameter space is a direct adjacent prior | 2025 hMC/ratchet/replica solution-space sampling and 2026 minibatch pseudo-Langevin Boltzmann sampling target broad solution-space exploration | [@Zambon2025SamplingSpace; @Zambon2026ControlledLangevin] | high | Related Work/Method | arXiv + PRE metadata |

## Enhanced Search Addendum

The enhanced sweep promoted `arXiv:2603.15367` through four independent paths: user seed, arXiv recency, lab-radar latest, and OpenAlex author latest. It also surfaced additional verified adjacent priors that narrow the safe novelty claim from "solution-space sampling" to "reference-local, QC-aware, rule-complexity-conditioned shell support profiling."

| claim | evidence | refs | confidence | insertion_point | basis |
| --- | --- | --- | --- | --- | --- |
| generic solution-space sampling is not novel enough | Boltzmann/Langevin, Gibbs posterior, and diffusion samplers already cover broad solution-space sampling questions | [@Zambon2025SamplingSpace; @Zambon2026ControlledLangevin; @Piccioli2024GibbsSampling; @Demyanenko2025GenerativeDiffusion] | high | Related Work/Limitation | enhanced sweep + DOI/arXiv metadata |
| connected solution-manifold theory is an active adjacent lineage | spherical negative perceptron and statistical-physics notes analyze solution manifolds and neighborhoods around solutions | [@Annesi2023StarShapedSpace; @Malatesta2023HighDimensionalManifold; @Baldassi2023TypicalAtypical] | high | Related Work | DOI/arXiv/OpenAlex metadata |
| temperature-based sampling is emerging beyond small feed-forward networks | feed-forward NN and protein-transformer studies both use temperature/Langevin views of parameter space | [@Zambon2026ControlledLangevin; @Ghiringhelli2026IntermediateTemperatures] | medium-high | Discussion | arXiv metadata |
| sampler-method claims and measurement-protocol claims must be separated | modern flows/diffusion/autoregressive samplers are studied as sampling algorithms in statistical-physics problems | [@Ghio2024SamplingFlows; @Demyanenko2025GenerativeDiffusion] | medium-high | Method/Discussion | DOI/arXiv metadata |
