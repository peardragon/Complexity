---
title: "Entropy-SGD: Biasing gradient descent into wide valleys"
tags: ["paper-card", "sota", "optimization", "local-entropy"]
aliases: ["Chaudhari2017EntropySGD", "Entropy-SGD: Biasing gradient descent into wide valleys"]
created: 2026-06-20
source: "curated bibliography; DOI/arXiv/OpenReview/PMLR/venue metadata"
confidence: high
---

# Entropy-SGD: Biasing gradient descent into wide valleys

## Metadata

- bibkey: [@Chaudhari2017EntropySGD]
- authors: Pratik Chaudhari, Anna Choromanska, Stefano Soatto, Yann LeCun, Carlo Baldassi, Christian Borgs, Jennifer Chayes, Levent Sagun, Riccardo Zecchina
- year: 2017
- venue: International Conference on Learning Representations
- DOI/arXiv/URL: n/a / 1611.01838 / https://arxiv.org/abs/1611.01838
- evidence_basis: metadata + abstract/PDF where available
- tags: sota, optimization, local-entropy

## Summary

inner-loop Langevin dynamics로 local entropy gradient를 추정해 wide valley를 선호하도록 SGD를 편향한다.

## Method

Local-entropy objective optimized by nested SGD/Langevin dynamics.

## Dataset

CNN/RNN benchmarks in the original experiments.

## Key Finding

local entropy objective가 smoother landscape와 generalization improvement를 보일 수 있다.

## Limitation

학습 알고리즘 제안이지, fixed reference 주변 phi(d) 곡선의 QC-aware 측정은 아니다.

## Relation To My Work

사용자의 방법은 optimization이 아니라 측정/진단에 초점을 둔다는 점에서 차별화된다.

## Cite For

local-entropy objective in deep-learning optimization.

## Backlinks

- [[02_Prior_Work_Map]]
- [[03_Novelty_Assessment]]
- [[maps/sota]] [[maps/optimization]]
