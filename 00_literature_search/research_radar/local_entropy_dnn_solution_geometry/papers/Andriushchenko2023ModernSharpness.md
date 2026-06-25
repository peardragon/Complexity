---
title: "A modern look at the relationship between sharpness and generalization"
tags: ["paper-card", "negative", "sharpness", "modern-settings"]
aliases: ["Andriushchenko2023ModernSharpness", "A modern look at the relationship between sharpness and generalization"]
created: 2026-06-20
source: "curated bibliography; DOI/arXiv/OpenReview/PMLR/venue metadata"
confidence: high
---

# A modern look at the relationship between sharpness and generalization

## Metadata

- bibkey: [@Andriushchenko2023ModernSharpness]
- authors: Maksym Andriushchenko, Francesco Croce, Maximilian Müller, Matthias Hein, Nicolas Flammarion
- year: 2023
- venue: International Conference on Machine Learning
- DOI/arXiv/URL: 10.5555/3618408.3618444 / 2302.07011 / https://arxiv.org/abs/2302.07011
- evidence_basis: metadata + abstract/PDF where available
- tags: negative, sharpness, modern-settings

## Summary

modern architectures/settings에서 sharpness-generalization correlation이 일관적이지 않음을 실험적으로 보였다.

## Method

Empirical study across ConvNets, transformers, fine-tuning settings.

## Dataset

ImageNet/CIFAR/CLIP/BERT style settings.

## Key Finding

sharpness can correlate with training hyperparameters and may fail as a universal explanation.

## Limitation

local volume/free energy curve와 dataset-rule complexity의 직접 분석은 아니다.

## Relation To My Work

사용자 논문이 과도한 generalization claim을 피해야 하는 핵심 근거다.

## Cite For

limitations of sharpness as universal generalization explanation.

## Backlinks

- [[02_Prior_Work_Map]]
- [[03_Novelty_Assessment]]
- [[maps/negative]] [[maps/sharpness]]
