---
title: "2026 Sampling Addendum"
tags: ["addendum", "2026", "sampling", "solution-space"]
aliases: ["Controlled Langevin addendum", "Zambon sampling addendum"]
created: 2026-06-22
source: "arXiv 2603.15367 + arXiv 2503.08266 + lab-radar"
confidence: high
---

# 10 2026 Sampling Addendum

## Verdict

Yes. [@Zambon2026ControlledLangevin] is a highly relevant 2026 addendum, and it has an even more directly solution-space-oriented 2025 predecessor [@Zambon2025SamplingSpace].

## Why It Matters

- It is from the Zecchina/Tiana/Malatesta line already visible in the lab-radar data.
- It targets sampling of neural-network parameter space according to a Boltzmann distribution, not just post-hoc visualization.
- It compares scalable minibatch pseudo-Langevin dynamics to exact hybrid Monte Carlo.
- It makes the "sampling solution space" prior-art threat stronger than the previous matrix indicated.

## How To Insert

| insertion_point | text | refs | caution |
| --- | --- | --- | --- |
| Related Work | 최근 Zambon 등은 feed-forward neural network의 parameter space를 Boltzmann distribution에 따라 직접 sampling하는 방법을 제안하며, exact hybrid Monte Carlo와 scalable pseudo-Langevin dynamics를 비교했다 [@Zambon2025SamplingSpace; @Zambon2026ControlledLangevin]. 이는 본 연구와 마찬가지로 solution-space geometry를 optimization 이후의 측정 대상으로 본다는 점에서 매우 가까운 prior art다. | [@Zambon2025SamplingSpace; @Zambon2026ControlledLangevin] | 본 연구의 차별점은 minibatch Langevin training/sampling이 아니라 reference-centered shell `phi(d)`와 rule-complexity-conditioned diagnostics임을 명시한다. |
| Method | 본 연구의 PM-SAIS/vMF shell estimator는 Boltzmann parameter-space sampler와 달리, 고정된 reference solution으로부터의 radius shell에서 support를 추정한다. 따라서 Zambon 등의 scalable Boltzmann sampling line과 비교할 때, sampling target과 geometry query가 다르다 [@Zambon2026ControlledLangevin]. | [@Zambon2026ControlledLangevin] | "우리가 더 scalable하다"는 식의 성능 비교는 하지 않는다. |
| Strongest Prior Art | 가장 강한 최신 prior art에는 density-of-states [@Mele2025DensityStates]뿐 아니라 neural-network solution-space sampling line [@Zambon2025SamplingSpace; @Zambon2026ControlledLangevin]도 포함되어야 한다. | [@Mele2025DensityStates; @Zambon2025SamplingSpace; @Zambon2026ControlledLangevin] | novelty defense를 "global/sampling method"가 아니라 "reference-local shell profile + QC + rule-family axis"로 둔다. |

## Claim Matrix Addendum

| claim | evidence | refs | confidence | insertion_point | basis |
| --- | --- | --- | --- | --- | --- |
| scalable sampling of NN parameter space is now a direct adjacent prior | 2025 hMC/ratchet/replica solution-space sampling and 2026 minibatch pseudo-Langevin Boltzmann sampling target the same broad space-exploration problem | [@Zambon2025SamplingSpace; @Zambon2026ControlledLangevin] | high | Related Work/Method | arXiv + PRE metadata |

## Novelty Adjustment

Before this addendum, the strongest adjacent measurement prior was framed mainly as global density-of-states [@Mele2025DensityStates]. After adding [@Zambon2025SamplingSpace; @Zambon2026ControlledLangevin], the novelty defense should explicitly say:

> 본 연구는 neural-network parameter space를 Boltzmann sampler로 전역 탐색하는 방법이 아니라, trained reference pool 주변의 radius shell에서 `phi(d)`와 QC diagnostics를 비교하는 reference-local measurement protocol이다.
