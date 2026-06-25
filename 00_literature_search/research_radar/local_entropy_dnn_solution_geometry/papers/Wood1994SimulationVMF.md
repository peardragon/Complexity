---
title: "Simulation of the von Mises Fisher distribution"
tags: ["paper-card", "method", "vmf", "directional-statistics"]
aliases: ["Wood1994SimulationVMF", "Simulation of the von Mises Fisher distribution"]
created: 2026-06-20
source: "curated bibliography; DOI/arXiv/OpenReview/PMLR/venue metadata"
confidence: high
---

# Simulation of the von Mises Fisher distribution

## Metadata

- bibkey: [@Wood1994SimulationVMF]
- authors: Andrew T. A. Wood
- year: 1994
- venue: Communications in Statistics - Simulation and Computation
- DOI/arXiv/URL: 10.1080/03610919408813161 / n/a / https://doi.org/10.1080/03610919408813161
- evidence_basis: metadata + abstract/PDF where available
- tags: method, vmf, directional-statistics

## Summary

hypersphere 위 von Mises-Fisher 분포 샘플링 알고리즘을 제시한다.

## Method

vMF random variate simulation.

## Dataset

Directional distributions on spheres.

## Key Finding

high-dimensional unit-vector proposals can be sampled from mean-direction concentrated distributions.

## Limitation

importance weighting/logZ estimation은 별도 설계가 필요하다.

## Relation To My Work

사용자 shell sampler의 vMF-centered proposal 구현 근거다.

## Cite For

vMF shell proposal sampling.

## Backlinks

- [[02_Prior_Work_Map]]
- [[03_Novelty_Assessment]]
- [[maps/method]] [[maps/vmf]]
