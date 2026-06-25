---
title: "Controlled Langevin Dynamics for Sampling of Feedforward Neural Networks Trained with Minibatches"
tags: ["paper", "solution-space", "langevin", "sampling", "minibatch", "2026"]
aliases: ["Zambon2026ControlledLangevin", "Controlled Langevin Dynamics"]
created: 2026-06-22
source: "arXiv metadata"
confidence: high
---

# Controlled Langevin Dynamics for Sampling of Feedforward Neural Networks Trained with Minibatches

- bibkey: [@Zambon2026ControlledLangevin]
- authors: Alessandro Zambon, Francesca Caruso, Riccardo Zecchina, Guido Tiana
- year: 2026
- venue: arXiv
- doi: 10.48550/arXiv.2603.15367
- arxiv: 2603.15367
- url: https://arxiv.org/abs/2603.15367

## Summary

The paper proposes pseudo-Langevin dynamics for Boltzmann sampling of feed-forward neural-network parameter space using minibatches in a controlled way. It compares equilibrium statistics against exact hybrid Monte Carlo, argues better scaling for large networks, and reports that intermediate temperatures can achieve strong generalization without validation-set early stopping.

## relation_to_my_work

This is highly relevant to the current project's sampling layer. It is not a local-entropy shell estimator, but it directly targets scalable sampling of neural-network solution space and therefore becomes a strong adjacent prior for the Method and Related Work sections.

## cite_for

- scalable minibatch Boltzmann/Langevin sampling of neural-network parameter space
- distinction between exact hMC and scalable stochastic dynamics
- strongest new adjacent prior for sampling-based solution-space exploration

## backlinks

- [[05_Claim_Evidence_Matrix]]
- [[10_2026_Sampling_Addendum]]
