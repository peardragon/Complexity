---
title: "Gibbs sampling the posterior of neural networks"
tags: ["paper", "gibbs-sampling", "posterior", "neural-networks", "2024"]
aliases: ["Piccioli2024GibbsSampling", "Gibbs sampling the posterior of neural networks"]
created: 2026-06-22
source: "enhanced sweep; DOI/lab-radar metadata"
confidence: high
---

# Gibbs sampling the posterior of neural networks

- bibkey: [@Piccioli2024GibbsSampling]
- authors: Giovanni Piccioli, Emanuele Troiani, Lenka Zdeborova
- year: 2024
- venue: Journal of Physics A: Mathematical and Theoretical
- doi: 10.1088/1751-8121/ad2c26
- url: https://doi.org/10.1088/1751-8121/ad2c26

## Summary

The paper proposes a probabilistic neural-network model with noise at pre- and post-activations and studies Gibbs sampling from the resulting posterior.

## Method

Gibbs sampler for a neural-network posterior, compared with MCMC baselines such as Hamiltonian Monte Carlo and MALA in small-model settings.

## Dataset

Synthetic and real data in small-model experiments, plus teacher-student framing.

## Key Finding

The method offers a sampling route for neural-network posterior distributions and includes a thermalization criterion in controlled settings.

## Limitation

It samples a posterior distribution, not a reference-centered shell support profile.

## Relation To My Work

This is a close adjacent sampling prior that should be used to clarify that the current project measures local shell geometry around trained references.

## Cite For

- neural-network posterior sampling prior art
- Gibbs/MCMC comparison in NN sampling context
- distinction between posterior sampling and local entropy shell profiling

## Backlinks

- [[11_Enhanced_Search_Update]]
- [[06_Strongest_Prior_Art]]
- [[maps/enhanced_sampling]]
