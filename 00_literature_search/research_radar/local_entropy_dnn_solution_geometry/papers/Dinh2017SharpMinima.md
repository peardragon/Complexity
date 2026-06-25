---
title: "Sharp minima can generalize for deep nets"
tags: ["paper-card", "limitation", "flatness", "reparameterization"]
aliases: ["Dinh2017SharpMinima", "Sharp minima can generalize for deep nets"]
created: 2026-06-20
source: "curated bibliography; DOI/arXiv/OpenReview/PMLR/venue metadata"
confidence: high
---

# Sharp minima can generalize for deep nets

## Metadata

- bibkey: [@Dinh2017SharpMinima]
- authors: Laurent Dinh, Razvan Pascanu, Samy Bengio, Yoshua Bengio
- year: 2017
- venue: International Conference on Machine Learning
- DOI/arXiv/URL: n/a / 1703.04933 / https://arxiv.org/abs/1703.04933
- evidence_basis: metadata + abstract/PDF where available
- tags: limitation, flatness, reparameterization

## Summary

deep nets에서는 flatness/sharpness measure가 reparameterization에 의해 조작될 수 있음을 보였다.

## Method

Analytical counterexamples based on parameter-space symmetries.

## Dataset

Deep ReLU network settings.

## Key Finding

naive parameter-space sharpness is not a reliable invariant explanation of generalization.

## Limitation

local entropy volume 자체를 완전히 부정하지는 않으며, 측정 좌표계와 regularization을 요구한다.

## Relation To My Work

사용자의 raw L2-shell 해석에서 반드시 언급해야 하는 주요 위협이다.

## Cite For

caveat about reparameterization-sensitive flatness.

## Backlinks

- [[02_Prior_Work_Map]]
- [[03_Novelty_Assessment]]
- [[maps/limitation]] [[maps/flatness]]
