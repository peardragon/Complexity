---
title: "Understanding deep learning requires rethinking generalization"
tags: ["paper-card", "seminal", "random-labels", "generalization"]
aliases: ["Zhang2017RethinkingGeneralization", "Understanding deep learning requires rethinking generalization"]
created: 2026-06-20
source: "curated bibliography; DOI/arXiv/OpenReview/PMLR/venue metadata"
confidence: high
---

# Understanding deep learning requires rethinking generalization

## Metadata

- bibkey: [@Zhang2017RethinkingGeneralization]
- authors: Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, Oriol Vinyals
- year: 2017
- venue: International Conference on Learning Representations
- DOI/arXiv/URL: n/a / 1611.03530 / https://arxiv.org/abs/1611.03530
- evidence_basis: metadata + abstract/PDF where available
- tags: seminal, random-labels, generalization

## Summary

deep nets가 random labels도 fit할 수 있음을 보여 일반화 설명에 데이터/알고리즘/implicit bias가 필요함을 제기했다.

## Method

Controlled label randomization experiments.

## Dataset

Image benchmarks with true/random labels.

## Key Finding

capacity alone cannot explain generalization.

## Limitation

random labels의 local entropy profile을 직접 측정하지 않는다.

## Relation To My Work

사용자 MNIST random_label rule이 왜 중요한 negative/control axis인지 설명한다.

## Cite For

random labels and memorization challenge.

## Backlinks

- [[02_Prior_Work_Map]]
- [[03_Novelty_Assessment]]
- [[maps/seminal]] [[maps/random_labels]]
