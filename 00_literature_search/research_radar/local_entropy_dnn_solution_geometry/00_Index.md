---
title: "Research Radar Index"
tags: ["index", "research-radar"]
aliases: ["local entropy DNN radar"]
created: 2026-06-20
source: "local code/results + megasearch + lab-radar"
confidence: high
---

# Research Radar: Local Entropy DNN Solution Geometry

## Inferred Project

로컬 `Complexity/local_project`의 promoted code/results를 기준으로, 현재 연구는 **Franz-Parisi/local-entropy 계열의 reference-centered shell free-energy/partition-function 측정법을 이론 perceptron에서 검증하고, 이를 3NN 및 MNIST rule-family DNN reference pools에 적용하여 dataset/rule complexity와 solution-space local support의 관계를 진단**하는 프로젝트로 확정했다.

## 핵심 결론 10개

1. local entropy는 spin-glass/perceptron의 constrained-overlap free energy에서 출발해 dense solution cluster를 찾는 도구로 발전했다 [@Franz1995RecipesMetastable; @Baldassi2016LocalEntropy].
2. Baldassi-Zecchina 계열은 rare but dense/wide regions가 학습 가능성과 robustness에 중요할 수 있음을 보였다 [@Baldassi2015SubdominantDense; @Baldassi2016UnreasonableEffectiveness; @Baldassi2020ShapingLandscape].
3. 사용자의 theory arm은 analytic full-RS perceptron curve와 shell PM-SAIS sampling curve를 비교하므로, empirical DNN 측정의 estimator validation layer로 쓰기 좋다 [@Gardner1988SpaceInteractions; @Seung1992StatisticalMechanics].
4. DNN/MNIST arm은 local entropy를 학습 알고리즘으로 쓰기보다, fixed reference 주변 `phi(d)` profile을 QC-aware sampling으로 측정한다는 점이 Entropy-SGD/SAM과 다르다 [@Chaudhari2017EntropySGD; @Foret2021SharpnessAware].
5. MNIST rule families에서 random label이 가장 큰 `-phi` magnitude를 보이고 structured rules가 낮게 묶인 로컬 결과는 dataset label complexity axis와 연결된다 [@Zhang2017RethinkingGeneralization; @Arpit2017CloserMemorization; @Shuman2013EmergingGraphSignal].
6. flatness/generalization을 직접 인과 주장하는 것은 위험하다. reparameterization, symmetry, modern sharpness correlation failures가 강한 반례다 [@Dinh2017SharpMinima; @Pittorino2022DeepNetworksToroids; @Andriushchenko2023ModernSharpness].
7. mode connectivity 문헌은 minima가 고립된 basin이라는 단순 그림을 약화시키므로, 사용자 결과는 “global disconnected basin”보다 “reference-local radial density/support”로 표현해야 한다 [@Garipov2018LossSurfaces; @Draxler2018EssentiallyNoBarriers].
8. 가장 가까운 최신 인접 prior art는 density-of-states/Wang-Landau식 global loss-spectrum 측정이며, 사용자 novelty는 global DoS가 아니라 reference-local shell curve와 MNIST rule-family axis다 [@Mele2025DensityStates].
9. sampling method 측면에서는 AIS/SMC/vMF 근거가 충분하지만, estimator QC와 split diagnostics를 본문에서 명시해야 한다 [@Neal2001AnnealedImportance; @DelMoral2006SMCSamplers; @Wood1994SimulationVMF].
10. 현재 MNIST 90-ref run은 complete이지만 QC diagnostic pass가 일부 반경에 한정되므로, 본문에서는 “diagnostic evidence”와 “promotion-ready claim”을 구분해야 한다.

## 다음 액션

- Introduction에서는 local entropy의 계보와 “왜 reference-local measurement가 필요한가”를 먼저 세운다.
- Related Work에서는 local entropy / flatness debate / mode connectivity / dataset complexity / sampling estimator를 분리해서 쓴다.
- Experiment에서는 theory validation과 MNIST diagnostic scope를 분리해 QC 기준을 전면에 둔다.
- Discussion에서는 generalization 인과가 아니라 “dataset/rule complexity와 local support geometry의 관찰적 연결”로 제한한다.

## 논문 본문에 바로 넣을 추천 문단 TOP 15

1. **Introduction**: 본 연구는 학습된 한 해 주변의 단순한 손실값이 아니라, reference parameter로부터의 거리 \(d\)에서 유지되는 해공간의 유효 부피를 측정 대상으로 삼는다. 이러한 관점은 고정된 reference와의 overlap을 제한한 자유에너지로 metastable structure를 해석하는 Franz-Parisi potential 및 local entropy 계보와 맞닿아 있다 [@Franz1995RecipesMetastable; @Baldassi2016LocalEntropy].
2. **Introduction**: 선행연구는 신경망의 해공간이 단순히 많은 isolated minima로만 구성되는 것이 아니라, 드물지만 조밀하고 접근 가능한 영역을 포함할 수 있음을 보여 왔다 [@Baldassi2015SubdominantDense; @Baldassi2016UnreasonableEffectiveness]. 따라서 특정 reference 주변의 local support profile을 직접 측정하는 것은 wide/robust solution hypothesis를 empirical하게 점검하는 한 방법이 된다.
3. **Related Work**: wide flat minima는 simple neural-network models에서 높은 margin 중심부와 그 주변의 dense solution structure로 해석되어 왔다 [@Baldassi2020ShapingLandscape; @Baldassi2021UnveilingStructure]. 본 연구는 이러한 이론적 그림을 전제로 삼되, 학습 알고리즘을 새로 제안하기보다 reference pool 주변의 \(\phi(d)\) 곡선을 추정하는 측정 문제로 재구성한다.
4. **Related Work**: Entropy-SGD와 SAM은 parameter neighborhood 정보를 학습 objective에 넣어 wide 혹은 sharpness-aware solution을 찾는 optimization 계열이다 [@Chaudhari2017EntropySGD; @Foret2021SharpnessAware]. 반면 본 연구의 중심은 optimization 성능 개선이 아니라, 이미 얻어진 reference solutions 주변에서 shell-wise partition estimate와 QC diagnostics를 통해 local geometry를 측정하는 것이다.
5. **Method**: 거리 shell에서의 partition-function 추정은 normalizing constant estimation 문제로 볼 수 있으며, annealed importance sampling과 SMC samplers는 이러한 분포열 기반 추정의 표준적 근거를 제공한다 [@Neal2001AnnealedImportance; @DelMoral2006SMCSamplers]. 본 구현은 hypersphere 방향 샘플링을 위해 von Mises-Fisher proposal을 사용하며, 이는 directional statistics의 표준 sampling scheme에 근거한다 [@Wood1994SimulationVMF].
6. **Method**: 이론 검증 단계에서는 perceptron local-entropy curve를 analytic full-RS baseline과 shell sampling estimate로 동시에 계산하여 estimator의 방향성을 점검한다. 이러한 설계는 perceptron solution-space를 통계물리적으로 분석한 고전 연구와 Franz-Parisi 계보를 DNN 측정으로 옮기기 전의 calibration layer로 기능한다 [@Gardner1988SpaceInteractions; @Seung1992StatisticalMechanics; @Franz1995RecipesMetastable].
7. **Experiment**: MNIST rule-family 실험은 true-structured, teacher-generated, low-TV, random-label 조건을 함께 두어 label complexity가 local support geometry에 미치는 영향을 관찰하도록 설계되었다. random labels가 capacity와 memorization 문제를 드러내는 강한 control임은 기존 연구에서 반복적으로 확인되었고 [@Zhang2017RethinkingGeneralization; @Arpit2017CloserMemorization], graph total variation은 data geometry 위 label smoothness를 표현하는 자연스러운 축이다 [@Shuman2013EmergingGraphSignal; @Ortega2018GraphSignalProcessing].
8. **Limitation**: parameter-space flatness는 reparameterization과 symmetry에 민감하므로, 본 연구의 L2-shell profile 역시 특정 architecture, regularization, coordinate convention 아래의 diagnostic quantity로 해석해야 한다 [@Dinh2017SharpMinima; @Pittorino2022DeepNetworksToroids]. 따라서 본문에서는 \(\phi(d)\)를 universal generalization measure가 아니라 fixed protocol에서의 reference-local support profile로 부른다.
9. **Discussion**: 최근의 대규모 sharpness 연구는 sharpness와 generalization의 관계가 architecture, hyperparameter, data setting에 따라 일관되지 않을 수 있음을 보였다 [@Andriushchenko2023ModernSharpness]. 본 연구의 기여는 이 논쟁을 우회하여, generalization을 직접 예측하기보다 dataset/rule condition에 따른 local support geometry의 변화를 측정하는 데 있다.
10. **Discussion**: mode connectivity 연구는 독립적으로 학습된 solutions가 low-loss curves로 연결될 수 있음을 보여, minima를 고립된 basin으로 보는 단순 그림을 약화시킨다 [@Garipov2018LossSurfaces; @Draxler2018EssentiallyNoBarriers]. 따라서 본 연구의 radial shell profile은 global connectivity의 대체물이 아니라, 특정 reference 주변의 local density/support를 보는 상보적 측정으로 해석된다.
11. **Related Work**: 최근에는 Wang-Landau sampling으로 neural network의 global density of states를 추정하고 dataset structure와 loss spectrum의 관계를 분석하는 연구도 등장했다 [@Mele2025DensityStates]. 본 연구는 이와 달리 전체 loss spectrum이 아니라 trained reference 주변의 거리별 shell profile을 추정하므로, global DoS와 reference-local local entropy를 구분해 비교한다.
12. **Related Work**: atypical high-margin solutions와 그 주변 local entropy는 binary/symmetric perceptron에서 최근까지 활발히 분석되고 있다 [@Baldassi2023TypicalAtypical; @Barbier2024AtypicalSolutions]. 이러한 결과는 효율적 알고리즘이 exponentially dominant typical solutions가 아니라 rare structured regions를 찾을 수 있다는 해석을 뒷받침한다.
13. **Discussion**: flat minima와 generalization의 formal link는 PAC-Bayes 관점에서도 연구되어 왔으며, 최근에는 gradient 및 functional inequality를 이용해 dimension-explicit dependence를 줄이는 bound가 제안되었다 [@Dziugaite2017ComputingNonvacuous; @Haddouche2025PACBayesianLink]. 다만 본 연구의 shell entropy는 bound 자체가 아니라 empirical diagnostic이므로, PAC-Bayes 연결은 해석적 가능성으로만 제시한다.
14. **Experiment**: 로컬 MNIST 결과는 90 references per rule 및 25개 radius grid의 mechanical sampling을 완료했지만, diagnostic QC pass는 일부 rule-radius에 제한되어 있다. 따라서 본문에서는 full grid를 exploratory measurement로, QC-passed subset을 stronger evidence로 구분해 보고한다.
15. **Introduction**: 요약하면, 본 연구의 기여는 local entropy를 학습 알고리즘이나 보편적 일반화 지표로 주장하는 것이 아니라, theory-validated shell estimator와 rule-complexity controls를 결합하여 DNN reference 주변 solution support를 측정하는 재현 가능한 protocol을 제시하는 데 있다 [@Baldassi2016LocalEntropy; @Neal2001AnnealedImportance; @Mele2025DensityStates].

## 반드시 읽어야 할 PDF TOP 20

1. [@Baldassi2016LocalEntropy] Local entropy as a measure for sampling solutions in constraint satisfaction problems (2016) — https://arxiv.org/pdf/1511.05634
2. [@Baldassi2015SubdominantDense] Subdominant dense clusters allow for simple learning and high computational performance in neural networks with discrete synapses (2015) — https://arxiv.org/pdf/1509.05753
3. [@Baldassi2016UnreasonableEffectiveness] Unreasonable effectiveness of learning neural networks: From accessible states and robust ensembles to basic algorithmic schemes (2016) — https://arxiv.org/pdf/1605.06444
4. [@Chaudhari2017EntropySGD] Entropy-SGD: Biasing gradient descent into wide valleys (2017) — https://arxiv.org/pdf/1611.01838
5. [@Baldassi2020ShapingLandscape] Shaping the learning landscape in neural networks around wide flat minima (2020) — https://arxiv.org/pdf/1905.07833
6. [@Baldassi2021UnveilingStructure] Unveiling the structure of wide flat minima in neural networks (2021) — https://arxiv.org/pdf/2107.01163
7. [@Pittorino2022DeepNetworksToroids] Deep networks on toroids: removing symmetries reveals the structure of flat regions in the landscape geometry (2022) — https://proceedings.mlr.press/v162/pittorino22a/pittorino22a.pdf
8. [@Baldassi2022LearningAtypical] Learning through atypical phase transitions in overparameterized neural networks (2022) — https://arxiv.org/pdf/2110.00683
9. [@Baldassi2023TypicalAtypical] Typical and atypical solutions in nonconvex neural networks with discrete and continuous weights (2023) — https://arxiv.org/pdf/2304.13871
10. [@Barbier2024AtypicalSolutions] On the atypical solutions of the symmetric binary perceptron (2024) — https://arxiv.org/pdf/2310.02850
11. [@Mele2025DensityStates] Density of states in neural networks: an in-depth exploration of learning in parameter space (2025) — https://openreview.net/pdf?id=BLDtWlFKhn
12. [@Dinh2017SharpMinima] Sharp minima can generalize for deep nets (2017) — https://arxiv.org/pdf/1703.04933
13. [@Andriushchenko2023ModernSharpness] A modern look at the relationship between sharpness and generalization (2023) — https://arxiv.org/pdf/2302.07011
14. [@Jiang2020FantasticGeneralization] Fantastic generalization measures and where to find them (2020) — https://arxiv.org/pdf/1912.02178
15. [@Zhang2017RethinkingGeneralization] Understanding deep learning requires rethinking generalization (2017) — https://arxiv.org/pdf/1611.03530
16. [@Arpit2017CloserMemorization] A closer look at memorization in deep networks (2017) — https://arxiv.org/pdf/1706.05394
17. [@Garipov2018LossSurfaces] Loss surfaces, mode connectivity, and fast ensembling of DNNs (2018) — https://arxiv.org/pdf/1802.10026
18. [@Neal2001AnnealedImportance] Annealed importance sampling (2001) — https://arxiv.org/pdf/physics/9803008
19. [@DelMoral2006SMCSamplers] Sequential Monte Carlo samplers (2006) — https://www.stats.ox.ac.uk/~doucet/delmoral_doucet_jasra_sequentialmontecarlosamplersJRSSB.pdf
20. [@Foret2021SharpnessAware] Sharpness-aware minimization for efficiently improving generalization (2021) — https://arxiv.org/pdf/2010.01412

## Vault Map

- [[01_Research_Landscape]]
- [[02_Prior_Work_Map]]
- [[03_Novelty_Assessment]]
- [[04_Implications]]
- [[05_Claim_Evidence_Matrix]]
- [[06_Strongest_Prior_Art]]
- [[07_Bibliography.bib]]

## Expanded 2026-06-22 Pass

- [[05_Claim_Evidence_Matrix]] now contains an expanded 60-reference claim/evidence matrix.
- [[08_Recent_Citation_Chase]] records top-prior-art citing/adjacent papers checked through OpenAlex plus DOI/arXiv verification.
- [[09_Expanded_Reference_List]] lists the 60 verified references used for manuscript citation planning.
- [[10_2026_Sampling_Addendum]] records the Zambon et al. 2025/2026 solution-space sampling addendum.
- [[11_Enhanced_Search_Update]] records the enhanced miss-prevention sweep that found and promoted `arXiv:2603.15367`, plus six additional verified adjacent sampling priors.
