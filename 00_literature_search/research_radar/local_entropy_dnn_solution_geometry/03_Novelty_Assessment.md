---
title: "Novelty Assessment"
tags: ["novelty", "assessment"]
aliases: ["novelty matrix"]
created: 2026-06-20
source: "local project evidence + curated literature"
confidence: high
---

# 03 Novelty Assessment

## Novelty Matrix

| 축 | 내 연구 | 가장 가까운 prior art | novelty 판단 |
| --- | --- | --- | --- |
| 문제정의 | reference-centered `phi(d)`/local support를 dataset/rule complexity와 연결 | dense cluster/wide minima theory [@Baldassi2020ShapingLandscape; @Baldassi2021UnveilingStructure] | **중-강**: DNN/MNIST rule-complexity axis가 다름 |
| 방법 | analytic perceptron validation + vMF/PM-SAIS shell estimator + QC diagnostics | local entropy Monte Carlo [@Baldassi2016LocalEntropy], AIS/SMC [@Neal2001AnnealedImportance; @DelMoral2006SMCSamplers] | **중**: estimator 조합/검증 workflow가 기여 |
| 데이터 | 3NN synthetic grid + MNIST binary rule families(low-TV/even-odd/teacher/random) | random-pattern perceptron, random labels [@Zhang2017RethinkingGeneralization] | **강**: local entropy profile을 rule complexity ladder에 직접 매핑 |
| 이론 | Full-RS perceptron baseline과 sampler convergence 비교 | perceptron statistical mechanics [@Gardner1988SpaceInteractions; @Seung1992StatisticalMechanics] | **중**: 새 이론보다 validation bridge |
| 평가 | QC pass, split diagnostics, bootstrap/ESS, reference-pool stability | generalization measure studies [@Jiang2020FantasticGeneralization] | **중-강**: measurement hygiene를 전면화 |
| 적용맥락 | MNIST reference family/local support diagnostic | density of states [@Mele2025DensityStates] | **중**: global DoS와 달리 reference-local profile |
| 한계극복 | flatness를 generalization causal claim이 아닌 diagnostic quantity로 제한 | sharpness critiques [@Dinh2017SharpMinima; @Andriushchenko2023ModernSharpness] | **강한 방어 논리** |

## 강한 novelty 후보

1. **이론-실험 bridge**: analytic perceptron local entropy와 shell estimator를 먼저 비교한 뒤 DNN/MNIST로 이식.
2. **dataset/rule complexity axis**: NMSTV/graph-TV 기반 rule ordering과 `phi(d)` profile을 연결.
3. **QC-aware local entropy measurement**: split/ESS/bootstrap diagnostics를 claim boundary로 사용.
4. **reference-family analysis**: single optimum이 아니라 trained references의 distribution/cluster를 분석.

## 약한 novelty 후보

- “flat minima가 generalization을 설명한다”는 주장은 기존 논쟁이 강하므로 novelty로 쓰면 위험하다 [@Dinh2017SharpMinima; @Andriushchenko2023ModernSharpness].
- “local entropy를 최적화한다”는 주장은 Entropy-SGD/entropic gradient prior와 겹친다 [@Chaudhari2017EntropySGD; @Pittorino2021EntropicGradient].
