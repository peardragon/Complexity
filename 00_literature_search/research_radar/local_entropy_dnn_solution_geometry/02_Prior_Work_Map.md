---
title: "Prior Work Map"
tags: ["prior-work", "references"]
aliases: ["prior work map"]
created: 2026-06-20
source: "curated verified references"
confidence: high
---

# 02 Prior Work Map

## 계보

Franz-Parisi constrained potential → perceptron/version-space statistical mechanics → local entropy/dense clusters → Entropy-SGD/entropic algorithms → wide-flat-minima structure → symmetry/flatness critique → density-of-states, posterior/Boltzmann sampling, and dataset-structure measurements.

## A → B → C Evolution

- **Problem evolution**: storage capacity and version space [@Gardner1988SpaceInteractions; @Seung1992StatisticalMechanics] → metastable/local free energy [@Franz1995RecipesMetastable] → dense solution clusters [@Baldassi2015SubdominantDense] → DNN wide minima and learning landscape [@Baldassi2020ShapingLandscape].
- **Method evolution**: replica/large-deviation theory → local entropy Monte Carlo [@Baldassi2016LocalEntropy] → Entropy-SGD [@Chaudhari2017EntropySGD] → posterior/Boltzmann solution-space samplers [@Piccioli2024GibbsSampling; @Zambon2025SamplingSpace; @Zambon2026ControlledLangevin] → SMC/vMF shell estimator in this project [@Neal2001AnnealedImportance; @DelMoral2006SMCSamplers; @Wood1994SimulationVMF].
- **Data evolution**: random pattern perceptron → true/noisy/random labels [@Zhang2017RethinkingGeneralization; @Arpit2017CloserMemorization] → graph-TV/NMSTV rule families in the local MNIST pipeline [@Shuman2013EmergingGraphSignal; @Ortega2018GraphSignalProcessing].

## 본문에 넣을 레퍼런스 목록

| 논점 | 넣을 레퍼런스 |
| --- | --- |
| local entropy의 원형 | [@Franz1995RecipesMetastable; @Baldassi2016LocalEntropy] |
| dense/wide solution regions | [@Baldassi2015SubdominantDense; @Baldassi2020ShapingLandscape; @Baldassi2021UnveilingStructure] |
| optimization과 구분 | [@Chaudhari2017EntropySGD; @Foret2021SharpnessAware; @Izmailov2018AveragingWeights] |
| flatness caveat | [@Dinh2017SharpMinima; @Pittorino2022DeepNetworksToroids; @Andriushchenko2023ModernSharpness] |
| global landscape/connectivity | [@Li2018VisualizingLoss; @Garipov2018LossSurfaces; @Draxler2018EssentiallyNoBarriers] |
| dataset complexity/control | [@Zhang2017RethinkingGeneralization; @Arpit2017CloserMemorization; @Shuman2013EmergingGraphSignal] |
| sampling/logZ estimator | [@Neal2001AnnealedImportance; @DelMoral2006SMCSamplers; @Wood1994SimulationVMF] |
| recent adjacent threat | [@Mele2025DensityStates; @Zambon2025SamplingSpace; @Zambon2026ControlledLangevin; @Piccioli2024GibbsSampling; @Demyanenko2025GenerativeDiffusion] |
| solution-manifold theory bridge | [@Annesi2023StarShapedSpace; @Malatesta2023HighDimensionalManifold; @Baldassi2023TypicalAtypical] |
