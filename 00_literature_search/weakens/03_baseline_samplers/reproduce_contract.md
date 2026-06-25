# Existing Method Reproduction Contract

The goal is clear, reproducible evidence about where existing methods fail on
this benchmark. The contract below is intentionally stricter than a figure-only
demo.

## HMC

Source paper: <https://arxiv.org/abs/2503.08266>

Proxy reproduction:

- Target: `pi(z) proportional to exp(-beta E(z))`.
- Potential: `U(z)=beta E(z)`.
- Integrator: leapfrog.
- Correction: Metropolis accept/reject using the exact Hamiltonian.
- Initialization: the `solution_core` center unless the config overrides it.
- Reported fields: step size, leapfrog steps, mass, acceptance rate, retained
  sample count, region hit counts, region mass error.

## pseudo-Langevin

Source paper: <https://arxiv.org/abs/2603.15367>

Proxy reproduction:

- Target: same `pi(z)`.
- Drift: gradient of the full smooth basin and fixed ridge terms plus minibatch
  estimates of rough dataset-gradient terms.
- Dynamics: underdamped Langevin with fixed friction and temperature `1/beta`.
- Initialization: the `solution_core` center unless the config overrides it.
- Reported fields: time step, friction, minibatch size, retained sample count,
  region hit counts, region mass error.

This is explicitly a proxy reproduction of the pL idea. A full paper-level pL
reimplementation should later replace this row if the project requires a
stronger empirical claim.

## Random-walk MCMC

Proxy reproduction:

- Target: same `pi(z)`.
- Proposal: isotropic Gaussian local proposal.
- Correction: Metropolis accept/reject.
- Reported fields: proposal scale, acceptance rate, retained sample count,
  region hit counts, region mass error.

