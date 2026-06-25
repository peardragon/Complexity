---
title: "Strongest Prior Art"
tags: ["prior-art", "threats"]
aliases: ["strongest prior art"]
created: 2026-06-20
source: "curated references"
confidence: high
---

# 06 Strongest Prior Art

| prior art | 왜 위협적인가 | 방어 논리 |
| --- | --- | --- |
| [@Baldassi2016LocalEntropy] | local entropy as sampling objective를 이미 제시 | 본 연구는 CSP solver가 아니라 DNN/MNIST reference-local shell profile + QC validation |
| [@Chaudhari2017EntropySGD] | local entropy를 DNN optimization에 적용 | 본 연구는 optimization이 아니라 measurement and diagnostics |
| [@Baldassi2020ShapingLandscape] | wide flat minima in neural networks와 가장 가까움 | simple model theory 중심; 본 연구는 rule complexity axis와 empirical reference-pool estimator |
| [@Baldassi2021UnveilingStructure] | wide flat minima structure 자체를 깊게 분석 | 사용자 결과는 그 구조를 empirical `phi(d)` profile로 관찰/검정하는 방향 |
| [@Mele2025DensityStates] | dataset structure와 parameter-space density를 직접 연결 | global density-of-states vs reference-centered radial local entropy; estimator와 claim이 다름 |
| [@Zambon2025SamplingSpace; @Zambon2026ControlledLangevin] | neural-network parameter space를 Boltzmann/hMC/pseudo-Langevin 방식으로 직접 sampling | 본 연구는 global sampler가 아니라 trained reference 주변 radius-shell support와 QC diagnostics |
| [@Piccioli2024GibbsSampling] | neural-network posterior sampling을 Gibbs sampler로 직접 다룸 | posterior distribution sampling과 fixed-reference shell profiling을 구분 |
| [@Annesi2023StarShapedSpace; @Demyanenko2025GenerativeDiffusion] | perceptron solution manifold와 generative sampling이 이미 active line | perceptron theory/sampler design과 DNN/MNIST rule-family diagnostic을 구분 |
| [@Dinh2017SharpMinima] | flatness 측정의 coordinate dependence를 공격 | limitation으로 수용하고, fixed architecture/regularized coordinate/QC diagnostic로 claim 범위 제한 |
| [@Andriushchenko2023ModernSharpness] | modern setting에서 sharpness-generalization 상관을 약화 | generalization causality 대신 local geometry measurement로 framing |
| [@Pittorino2022DeepNetworksToroids] | symmetry 제거 필요성을 제기 | raw L2 distance caveat를 명시하고 future work로 quotient/invariant distance 제안 |
| [@Zhang2017RethinkingGeneralization] | random labels에서도 fit 가능함을 보여 단순 capacity/flatness 설명을 약화 | random_label을 control로 포함하여 local support profile 차이를 직접 측정 |
| [@Garipov2018LossSurfaces] | minima가 low-loss paths로 연결될 수 있음 | global connectivity와 reference-local radial density는 상보적 quantity라고 방어 |
