---
title: "Implications"
tags: ["implications", "claims"]
aliases: ["claim boundaries"]
created: 2026-06-20
source: "curated literature + local reports"
confidence: high
---

# 04 Implications

## 주장 가능한 문장

- 본 연구는 local entropy를 generalization의 단일 원인으로 주장하기보다, fixed reference 주변의 radial support/free-energy profile을 측정하는 진단 도구로 사용한다 [@Dinh2017SharpMinima; @Jiang2020FantasticGeneralization; @Andriushchenko2023ModernSharpness].
- local entropy/dense cluster 계열은 simple neural models에서 rare but wide regions가 알고리즘적으로 중요할 수 있음을 보여 왔고, 본 연구는 그 관점을 DNN/MNIST reference-family measurement로 확장한다 [@Baldassi2015SubdominantDense; @Baldassi2020ShapingLandscape; @Baldassi2021UnveilingStructure].
- random-label 및 structured-label controls는 dataset complexity가 training behavior와 solution geometry를 바꿀 수 있음을 검토하는 자연스러운 실험축이다 [@Zhang2017RethinkingGeneralization; @Arpit2017CloserMemorization; @Shuman2013EmergingGraphSignal].

## 주장하면 안 되는 문장

- “local entropy가 높으면 항상 generalization이 좋다.” → reparameterization, data dependence, modern sharpness counter-evidence 때문에 과장이다 [@Dinh2017SharpMinima; @Andriushchenko2023ModernSharpness].
- “MNIST 90-ref 결과는 모든 반경에서 QC-validated이다.” → local report상 diagnostic QC pass가 제한적이므로 불가.
- “global loss landscape의 모든 density of states를 측정했다.” → 본 연구는 reference-centered shell profile이며 global DoS와 다르다 [@Mele2025DensityStates].

## 학술적/방법론적 함의

- 학술적으로는 local entropy theory를 empirical DNN diagnostic protocol로 번역한다.
- 방법론적으로는 logZ/partition-function estimator와 QC criteria를 함께 제시해 후속 연구가 재현 가능한 local geometry measurement를 할 수 있게 한다 [@Neal2001AnnealedImportance; @DelMoral2006SMCSamplers].
- 실무적으로는 dataset/rule complexity가 trained solution support profile에 미치는 영향을 진단하는 lightweight benchmark를 제안할 수 있다.
