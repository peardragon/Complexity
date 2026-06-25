# Baseline Notes

## HMC / RHMC paper

Source: <https://arxiv.org/abs/2503.08266>

Zambon, Malatesta, Tiana, and Zecchina study the solution space of a one-hidden
layer artificial neural network using hybrid Monte Carlo, a ratchet variant,
and coupled replicas. The arXiv abstract emphasizes that near the interpolation
threshold the low-energy manifold can become spiky, while in an
overparameterized regime it becomes flatter and easier to sample.

Benchmark interpretation:

- Use full-gradient HMC as the exact local-chain reference.
- Do not claim HMC is incorrect.
- Test a difficult landscape in which low-loss regions are separated by narrow
  necks or high barriers, so finite-budget chains started at one solution fail
  to discover all relevant regions.

## pseudo-Langevin paper

Source: <https://arxiv.org/abs/2603.15367>

Zambon, Caruso, Zecchina, and Tiana propose controlled minibatch
pseudo-Langevin dynamics for Boltzmann sampling of feed-forward neural networks.
The arXiv abstract frames it as a scalable minibatch alternative to exact HMC,
with controlled gradient-noise statistics.

Benchmark interpretation:

- Use a minibatch underdamped Langevin proxy with noisy dataset-gradient terms.
- Do not claim this is a line-by-line reproduction of the paper.
- Test whether minibatch diffusion alone discovers all disconnected or
  needle-like low-loss regions under a fixed compute budget.

## vMF + L2 recovery baseline

The recovery method is deliberately global: inspect the proxy landscape, choose
region directions around the solution, sample directions with vMF proposals and
radii with L2 shells, then self-normalize importance weights against the target
Boltzmann density.

This is aligned with the `../../02_dnn/04_sampling` vocabulary: shell sampling,
vMF tilt, importance weights, ESS fraction, weighted region ratios, and QC.

