---
title: "Generative diffusion for perceptron problems: statistical physics analysis and efficient algorithms"
tags: ["paper", "perceptron", "diffusion", "sampling", "solution-space", "2025"]
aliases: ["Demyanenko2025GenerativeDiffusion", "Generative diffusion for perceptron problems"]
created: 2026-06-22
source: "enhanced sweep; arXiv/OpenAlex/lab-radar metadata"
confidence: high
---

# Generative diffusion for perceptron problems

- bibkey: [@Demyanenko2025GenerativeDiffusion]
- authors: Elizaveta Demyanenko, Davide Straziota, Carlo Baldassi, Carlo Lucibello
- year: 2025
- venue: arXiv
- doi: 10.48550/arXiv.2502.16292
- arxiv: 2502.16292
- url: https://arxiv.org/abs/2502.16292

## Summary

The paper analyzes efficient sampling of nonconvex perceptron solution spaces using generative diffusion algorithms, with replica-theory predictions and algorithmic comparisons across spherical and binary perceptron settings.

## Method

Replica-theory analysis of diffusion-based sampling, with Approximate Message Passing used as an idealized score-function reference.

## Dataset

Random high-dimensional perceptron instances.

## Key Finding

Diffusion-style generative samplers can efficiently sample broad regions of the spherical perceptron solution space in much of the replica-symmetric regime, while binary weights remain harder.

## Limitation

This is a perceptron/statistical-mechanics model, not an empirical DNN/MNIST reference-pool measurement.

## Relation To My Work

It strengthens the adjacent prior-art line around sampling solution spaces, but it does not replace the current project's reference-local shell `phi(d)` diagnostic.

## Cite For

- diffusion/generative sampling as a recent solution-space method
- theoretical bridge between perceptron solution-space geometry and sampler design
- caution against claiming novelty as generic "solution-space sampling"

## Backlinks

- [[11_Enhanced_Search_Update]]
- [[05_Claim_Evidence_Matrix]]
- [[maps/enhanced_sampling]]
