---
title: "Recent Citation Chase"
tags: ["citation-chase", "recent", "openalex"]
aliases: ["recent citing papers"]
created: 2026-06-22
source: "OpenAlex recent citers + arXiv/DOI/web verification"
confidence: medium-high
---

# 08 Recent Citation Chase

## Method

Top prior art seeds were searched through OpenAlex citing-work queries, then filtered by topic relevance and DOI/arXiv/official venue verification. Raw API output is stored at `megasearch/openalex_recent_citers.json`; only verified items are used below.

| seed/reference family | verified recent citing or adjacent papers | interpretation |
| --- | --- | --- |
| [@Baldassi2016LocalEntropy] | [@Musso2021PartialLocalEntropy; @Abbe2022BinaryPerceptron; @Baldassi2023TypicalAtypical; @Catania2024CopycatPerceptron; @Barbier2025EscapeAtypical; @Benedetti2025OverlapGap] | local entropy는 DNN weight anisotropy와 perceptron connected/atypical solution theory로 확장됐다. 본 연구는 이 흐름을 DNN rule-family shell measurement로 옮기는 위치다. |
| [@Chaudhari2017EntropySGD] | [@Kwon2021ASAM; @Liu2022EfficientSAM; @Zhang2023GradientNormAware; @Wu2024CRSAM] | 최근 인용 흐름은 대부분 optimizer 개선이다. 따라서 본 연구가 optimizer 성능 개선이 아니라 measurement protocol임을 강하게 분리해야 한다. |
| [@Dinh2017SharpMinima; @Jiang2020FantasticGeneralization] | [@Zhang2021WhyFlatness; @Kwon2021ASAM; @Yang2021TaxonomizingLandscape; @Andriushchenko2023ModernSharpness; @Abdollahpoorrostam2024CLIPSharpness] | 최근 연구는 flatness metric의 취약성과 조건부 유효성을 동시에 보여준다. phi(d)는 universal generalization predictor가 아니라 fixed protocol diagnostic으로 써야 한다. |
| [@Garipov2018LossSurfaces; @Draxler2018EssentiallyNoBarriers] | [@Entezari2022PermutationModeConnectivity; @Ainsworth2023GitReBasin; @Abdollahpoorrostam2024CLIPSharpness] | mode connectivity는 symmetry quotient와 interpolation 관점으로 발전했다. 사용자의 radial shell profile은 global basin connectivity claim의 대체물이 아니다. |
| [@Zhang2017RethinkingGeneralization] | [@Nakkiran2020DeepDoubleDescent; @Belkin2019ReconcilingBiasVariance; @Goldt2020HiddenManifold; @Mele2025DensityStates] | random-label control은 단순 stress test에서 dataset structure/complexity와 loss-space volume을 연결하는 흐름으로 확장됐다. |
| [@Mele2025DensityStates] | [@Wang2001FlatHistogram; @Liu2023GradientWangLandau; @Winer2026DeepNeuralNetsHamiltonians; @Ly2025MultifractalLandscapes] | Mele는 아직 너무 최근이라 citing corpus가 작다. 대신 Wang-Landau/DOS lineage와 2025 loss-landscape model을 함께 써 adjacent prior로 다룬다. |

## Strongest Recent Signals

1. **Local entropy/perceptron line is active through 2025**: rare connected clusters, atypical regions, and overlap-gap thresholds remain central [@Abbe2022BinaryPerceptron; @Barbier2025EscapeAtypical; @Benedetti2025OverlapGap].
2. **Sharpness line is crowded on optimization**: ASAM, efficient SAM, GNAM, and CR-SAM make optimizer novelty hard to claim [@Kwon2021ASAM; @Liu2022EfficientSAM; @Zhang2023GradientNormAware; @Wu2024CRSAM].
3. **Flatness as universal predictor is weak**: recent work emphasizes metric dependence, architecture dependence, and local/global distinctions [@Zhang2021WhyFlatness; @Yang2021TaxonomizingLandscape; @Andriushchenko2023ModernSharpness; @Abdollahpoorrostam2024CLIPSharpness].
4. **Density-of-states is the closest adjacent measurement program**: it should be framed as global DOS, while the current project is reference-local shell entropy [@Wang2001FlatHistogram; @Liu2023GradientWangLandau; @Mele2025DensityStates].
