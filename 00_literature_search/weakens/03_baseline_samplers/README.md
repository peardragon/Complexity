# Baseline Samplers

Implemented finite-budget baselines:

- `random_walk_mcmc`: local Gaussian random-walk Metropolis.
- `hmc`: full-gradient Hamiltonian Monte Carlo on the proxy energy.
- `pseudo_langevin`: minibatch underdamped Langevin proxy for pL-like dynamics.

The comparison is intentionally about finite-budget reachability and sample
quality in a difficult landscape. It is not a proof that the algorithms fail in
all parameterizations.

## Internal reproduction mapping

| Method | Source role | Proxy implementation | Required diagnostics |
| --- | --- | --- | --- |
| HMC | Full-gradient exact sampler from arXiv:2503.08266. | Leapfrog HMC on `U(z)=beta E(z)` with Metropolis correction. | Acceptance rate, retained samples, region hit counts, region mass error. |
| pL | Minibatch controlled Langevin sampler from arXiv:2603.15367. | Underdamped minibatch Langevin using stochastic rough-term gradients and fixed friction. | Retained samples, diffusion scale, region hit counts, region mass error. |
| MCMC | Simple local-chain control. | Gaussian random-walk Metropolis on `U(z)=beta E(z)`. | Acceptance rate, retained samples, region hit counts, region mass error. |

The pL row is a proxy reproduction, not a full implementation of the paper's
all-parameter control scheme. The output JSON records this distinction so the
claim remains explicit.
