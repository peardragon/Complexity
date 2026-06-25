---
title: "Computing nonvacuous generalization bounds for deep (stochastic) neural networks with many more parameters than training data"
tags: ["paper-card", "pac-bayes", "generalization", "flatness"]
aliases: ["Dziugaite2017ComputingNonvacuous", "Computing nonvacuous generalization bounds for deep (stochastic) neural networks with many more parameters than training data"]
created: 2026-06-20
source: "curated bibliography; DOI/arXiv/OpenReview/PMLR/venue metadata"
confidence: high
---

# Computing nonvacuous generalization bounds for deep (stochastic) neural networks with many more parameters than training data

## Metadata

- bibkey: [@Dziugaite2017ComputingNonvacuous]
- authors: Gintare Karolina Dziugaite, Daniel M. Roy
- year: 2017
- venue: Conference on Uncertainty in Artificial Intelligence
- DOI/arXiv/URL: n/a / 1703.11008 / https://arxiv.org/abs/1703.11008
- evidence_basis: metadata + abstract/PDF where available
- tags: pac-bayes, generalization, flatness

## Summary

PAC-Bayes posterior를 학습해 overparameterized net에 nonvacuous bound를 계산했다.

## Method

PAC-Bayes bound optimization.

## Dataset

MNIST-style experiments.

## Key Finding

stochastic neural-network posterior와 generalization bounds를 실제로 계산할 수 있음을 보였다.

## Limitation

local radial density curve를 직접 측정하지 않는다.

## Relation To My Work

사용자 MNIST local entropy를 generalization bound와 연결할 때 참고하되, 동일 주장으로 과장하면 안 된다.

## Cite For

PAC-Bayes posterior/flatness relation.

## Backlinks

- [[02_Prior_Work_Map]]
- [[03_Novelty_Assessment]]
- [[maps/pac_bayes]] [[maps/generalization]]
