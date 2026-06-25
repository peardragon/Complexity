---
title: "Research Landscape"
tags: ["landscape", "literature"]
aliases: ["research landscape"]
created: 2026-06-20
source: "curated verified references + megasearch corpus"
confidence: high
---

# 01 Research Landscape

## 현재 연구지형

local entropy 연구는 세 갈래로 나뉜다. 첫째, Franz-Parisi potential과 perceptron/CSP 해공간의 dense cluster 계열은 reference 주변 constrained free energy를 통해 rare but accessible solution regions를 해석한다 [@Franz1995RecipesMetastable; @Baldassi2015SubdominantDense; @Baldassi2016LocalEntropy]. 둘째, deep learning에서는 Entropy-SGD, SAM, SWA처럼 neighborhood geometry를 학습 알고리즘이나 regularizer로 쓰는 흐름이 있다 [@Chaudhari2017EntropySGD; @Foret2021SharpnessAware; @Izmailov2018AveragingWeights]. 셋째, sharpness/flatness 자체가 generalization explanation으로 충분한지에 대한 강한 회의가 존재한다 [@Dinh2017SharpMinima; @Jiang2020FantasticGeneralization; @Andriushchenko2023ModernSharpness].

## 연도별 흐름

| 기간 | 흐름 | 핵심 레퍼런스 |
| --- | --- | --- |
| 1988–1995 | perceptron capacity, version space, Franz-Parisi potential | [@Gardner1988SpaceInteractions; @Seung1992StatisticalMechanics; @Franz1995RecipesMetastable] |
| 2015–2017 | local entropy/dense clusters and Entropy-SGD | [@Baldassi2015SubdominantDense; @Baldassi2016LocalEntropy; @Chaudhari2017EntropySGD] |
| 2017–2020 | random labels, flatness critique, mode connectivity, generalization measures | [@Zhang2017RethinkingGeneralization; @Dinh2017SharpMinima; @Garipov2018LossSurfaces; @Jiang2020FantasticGeneralization] |
| 2020–2023 | wide flat minima structure, symmetry-aware geometry, SAM/PAC-Bayes debate | [@Baldassi2020ShapingLandscape; @Baldassi2021UnveilingStructure; @Pittorino2022DeepNetworksToroids; @Foret2021SharpnessAware] |
| 2024–2026 | atypical perceptron solutions, density-of-states, renewed statistical-physics framing | [@Barbier2024AtypicalSolutions; @Mele2025DensityStates; @Winer2026DeepNeuralNetsHamiltonians] |

## 주요 클러스터

- **Statistical physics local entropy**: Franz-Parisi, perceptron, robust ensembles, atypical high-margin clusters.
- **Optimization toward wide regions**: Entropy-SGD, entropic gradient descent, SAM, SWA.
- **Measurement and geometry**: loss landscape visualization, mode connectivity, density of states.
- **Dataset/rule complexity**: random labels, memorization, graph total variation / label smoothness.
- **Estimator technology**: AIS/SMC/vMF shell sampling and log normalizer estimation.

## Saturation / 전환

flatness-generalization의 단순 상관 주장은 이미 포화와 반례가 많다 [@Dinh2017SharpMinima; @Andriushchenko2023ModernSharpness]. 반면 **어떤 coordinate/metric/reference/dataset 조건에서 어떤 local geometry가 관찰되는가**는 여전히 열려 있으며, density-of-states나 reference-local shell profile 같은 측정형 연구가 새롭게 부상 중이다 [@Mele2025DensityStates].
