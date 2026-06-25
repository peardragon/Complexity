---
title: "Averaging weights leads to wider optima and better generalization"
tags: ["paper-card", "optimization", "flatness", "swa"]
aliases: ["Izmailov2018AveragingWeights", "Averaging weights leads to wider optima and better generalization"]
created: 2026-06-20
source: "curated bibliography; DOI/arXiv/OpenReview/PMLR/venue metadata"
confidence: high
---

# Averaging weights leads to wider optima and better generalization

## Metadata

- bibkey: [@Izmailov2018AveragingWeights]
- authors: Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, Andrew Gordon Wilson
- year: 2018
- venue: Conference on Uncertainty in Artificial Intelligence
- DOI/arXiv/URL: n/a / 1803.05407 / https://arxiv.org/abs/1803.05407
- evidence_basis: metadata + abstract/PDF where available
- tags: optimization, flatness, swa

## Summary

stochastic weight averaging이 wider optima와 더 나은 generalization을 유도할 수 있음을 보였다.

## Method

Weight averaging along SGD trajectory.

## Dataset

Vision benchmarks.

## Key Finding

SGD trajectory samples can be averaged to land in flatter regions.

## Limitation

volume profile의 absolute partition function 추정은 아니다.

## Relation To My Work

local entropy measurement와 optimization trajectory-based flatness를 구분한다.

## Cite For

wider optima through stochastic weight averaging.

## Backlinks

- [[02_Prior_Work_Map]]
- [[03_Novelty_Assessment]]
- [[maps/optimization]] [[maps/flatness]]
