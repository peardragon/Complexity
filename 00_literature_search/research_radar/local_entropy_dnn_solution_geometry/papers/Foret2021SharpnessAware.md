---
title: "Sharpness-aware minimization for efficiently improving generalization"
tags: ["paper-card", "sota", "sharpness", "optimization"]
aliases: ["Foret2021SharpnessAware", "Sharpness-aware minimization for efficiently improving generalization"]
created: 2026-06-20
source: "curated bibliography; DOI/arXiv/OpenReview/PMLR/venue metadata"
confidence: high
---

# Sharpness-aware minimization for efficiently improving generalization

## Metadata

- bibkey: [@Foret2021SharpnessAware]
- authors: Pierre Foret, Ariel Kleiner, Hossein Mobahi, Behnam Neyshabur
- year: 2021
- venue: International Conference on Learning Representations
- DOI/arXiv/URL: n/a / 2010.01412 / https://arxiv.org/abs/2010.01412
- evidence_basis: metadata + abstract/PDF where available
- tags: sota, sharpness, optimization

## Summary

parameter neighborhood의 worst-case loss를 줄이는 SAM objective를 제안했다.

## Method

Min-max sharpness-aware training objective.

## Dataset

Vision/NLP benchmark experiments.

## Key Finding

neighborhood-aware optimization can improve generalization in many settings.

## Limitation

SAM은 training objective이며 local entropy density profile estimation과 다르다.

## Relation To My Work

사용자 방법을 optimization method가 아닌 measurement protocol로 구분할 때 필요하다.

## Cite For

neighborhood-aware optimization baseline.

## Backlinks

- [[02_Prior_Work_Map]]
- [[03_Novelty_Assessment]]
- [[maps/sota]] [[maps/sharpness]]
