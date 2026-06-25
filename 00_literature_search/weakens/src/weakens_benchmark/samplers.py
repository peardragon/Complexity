from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .landscape import ProxyLandscape


@dataclass
class SampleResult:
    name: str
    samples: np.ndarray
    energies: np.ndarray
    accept_rate: float | None
    metadata: dict[str, Any]


def _retain(step: int, burn_in: int, thin: int) -> bool:
    return step >= burn_in and ((step - burn_in) % thin == 0)


def _clip_reflect(
    z: np.ndarray,
    v: np.ndarray,
    landscape: ProxyLandscape,
) -> tuple[np.ndarray, np.ndarray]:
    out = z.copy()
    vel = v.copy()
    if out[0] < landscape.xlim[0] or out[0] > landscape.xlim[1]:
        out[0] = float(np.clip(out[0], landscape.xlim[0], landscape.xlim[1]))
        vel[0] *= -0.5
    if out[1] < landscape.ylim[0] or out[1] > landscape.ylim[1]:
        out[1] = float(np.clip(out[1], landscape.ylim[0], landscape.ylim[1]))
        vel[1] *= -0.5
    return out, vel


def run_random_walk(
    landscape: ProxyLandscape,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> SampleResult:
    sampler_cfg = config["samplers"]
    rw_cfg = sampler_cfg["random_walk"]
    n_steps = int(sampler_cfg["n_steps"])
    burn_in = int(sampler_cfg["burn_in"])
    thin = int(sampler_cfg["thin"])
    scale = float(rw_cfg["proposal_scale"])
    current = np.asarray(sampler_cfg["initial"], dtype=np.float64)
    current_energy = float(landscape.energy(current)[0])
    samples = []
    energies = []
    accepted = 0
    for step in range(n_steps):
        proposal = current + rng.normal(scale=scale, size=2)
        proposal_energy = float(landscape.energy(proposal)[0])
        log_accept = -landscape.beta * (proposal_energy - current_energy)
        if np.log(rng.random()) <= min(0.0, log_accept):
            current = proposal
            current_energy = proposal_energy
            accepted += 1
        if _retain(step, burn_in, thin):
            samples.append(current.copy())
            energies.append(current_energy)
    return SampleResult(
        name="random_walk_mcmc",
        samples=np.asarray(samples, dtype=np.float64),
        energies=np.asarray(energies, dtype=np.float64),
        accept_rate=accepted / max(1, n_steps),
        metadata={
            "method_role": "existing_method_reproduction",
            "source_paper": "local random-walk control",
            "proxy_mapping": "Gaussian random-walk Metropolis on U(z)=beta*E(z)",
            "proposal_scale": scale,
            "n_steps": n_steps,
            "burn_in": burn_in,
            "thin": thin,
        },
    )


def _hmc_propose(
    landscape: ProxyLandscape,
    current: np.ndarray,
    step_size: float,
    leapfrog_steps: int,
    mass: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, bool]:
    q = current.copy()
    p = rng.normal(scale=np.sqrt(mass), size=2)
    current_p = p.copy()
    current_e = float(landscape.energy(q)[0])
    current_h = landscape.beta * current_e + 0.5 * float(np.dot(current_p, current_p)) / mass
    grad = landscape.grad(q)[0] * landscape.beta
    p -= 0.5 * step_size * grad
    valid = True
    for i in range(leapfrog_steps):
        q = q + step_size * p / mass
        if not landscape.inside_domain(q):
            valid = False
            break
        grad = landscape.grad(q)[0] * landscape.beta
        if i != leapfrog_steps - 1:
            p -= step_size * grad
    if valid:
        p -= 0.5 * step_size * grad
        p = -p
        proposal_e = float(landscape.energy(q)[0])
        proposal_h = landscape.beta * proposal_e + 0.5 * float(np.dot(p, p)) / mass
        log_accept = current_h - proposal_h
        if np.isfinite(log_accept) and np.log(rng.random()) <= min(0.0, log_accept):
            return q, True
    return current, False


def run_hmc(
    landscape: ProxyLandscape,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> SampleResult:
    sampler_cfg = config["samplers"]
    hmc_cfg = sampler_cfg["hmc"]
    n_steps = int(sampler_cfg["n_steps"])
    burn_in = int(sampler_cfg["burn_in"])
    thin = int(sampler_cfg["thin"])
    step_size = float(hmc_cfg["step_size"])
    leapfrog_steps = int(hmc_cfg["leapfrog_steps"])
    mass = float(hmc_cfg["mass"])
    current = np.asarray(sampler_cfg["initial"], dtype=np.float64)
    samples = []
    energies = []
    accepted = 0
    for step in range(n_steps):
        current, did_accept = _hmc_propose(
            landscape,
            current,
            step_size,
            leapfrog_steps,
            mass,
            rng,
        )
        accepted += int(did_accept)
        if _retain(step, burn_in, thin):
            samples.append(current.copy())
            energies.append(float(landscape.energy(current)[0]))
    return SampleResult(
        name="hmc",
        samples=np.asarray(samples, dtype=np.float64),
        energies=np.asarray(energies, dtype=np.float64),
        accept_rate=accepted / max(1, n_steps),
        metadata={
            "method_role": "existing_method_reproduction",
            "source_paper": "arXiv:2503.08266",
            "proxy_mapping": "full-gradient leapfrog HMC on U(z)=beta*E(z)",
            "step_size": step_size,
            "leapfrog_steps": leapfrog_steps,
            "mass": mass,
            "n_steps": n_steps,
            "burn_in": burn_in,
            "thin": thin,
        },
    )


def run_pseudo_langevin(
    landscape: ProxyLandscape,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> SampleResult:
    sampler_cfg = config["samplers"]
    pl_cfg = sampler_cfg["pseudo_langevin"]
    n_steps = int(sampler_cfg["n_steps"])
    burn_in = int(sampler_cfg["burn_in"])
    thin = int(sampler_cfg["thin"])
    dt = float(pl_cfg["dt"])
    friction = float(pl_cfg["friction"])
    batch_size = int(pl_cfg["batch_size"])
    current = np.asarray(sampler_cfg["initial"], dtype=np.float64)
    velocity = np.zeros(2, dtype=np.float64)
    samples = []
    energies = []
    n_terms = max(1, landscape.rough_count)
    noise_scale = np.sqrt(2.0 * friction * dt / landscape.beta)
    for step in range(n_steps):
        if landscape.rough_count > 0:
            idx = rng.choice(n_terms, size=min(batch_size, n_terms), replace=False)
        else:
            idx = None
        grad_e = landscape.grad(current, rough_batch_indices=idx)[0]
        velocity = (1.0 - friction * dt) * velocity - dt * grad_e
        velocity += noise_scale * rng.normal(size=2)
        proposal = current + dt * velocity
        current, velocity = _clip_reflect(proposal, velocity, landscape)
        if _retain(step, burn_in, thin):
            samples.append(current.copy())
            energies.append(float(landscape.energy(current)[0]))
    return SampleResult(
        name="pseudo_langevin",
        samples=np.asarray(samples, dtype=np.float64),
        energies=np.asarray(energies, dtype=np.float64),
        accept_rate=None,
        metadata={
            "method_role": "existing_method_proxy_reproduction",
            "source_paper": "arXiv:2603.15367",
            "proxy_mapping": "underdamped minibatch Langevin with rough-term gradient minibatches",
            "dt": dt,
            "friction": friction,
            "batch_size": batch_size,
            "n_steps": n_steps,
            "burn_in": burn_in,
            "thin": thin,
        },
    )


def run_all_baselines(
    landscape: ProxyLandscape,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, SampleResult]:
    seeds = rng.integers(0, 2**32 - 1, size=3, dtype=np.uint32)
    results = {
        "random_walk_mcmc": _timed_sampler(
            run_random_walk,
            landscape,
            config,
            np.random.default_rng(int(seeds[0])),
        ),
        "hmc": _timed_sampler(
            run_hmc,
            landscape,
            config,
            np.random.default_rng(int(seeds[1])),
        ),
        "pseudo_langevin": _timed_sampler(
            run_pseudo_langevin,
            landscape,
            config,
            np.random.default_rng(int(seeds[2])),
        ),
    }
    return results


def _timed_sampler(
    fn: Any,
    landscape: ProxyLandscape,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> SampleResult:
    start = time.perf_counter()
    result = fn(landscape, config, rng)
    elapsed = time.perf_counter() - start
    result.metadata = result.metadata | {
        "attempted_sample_count": int(config["samplers"]["n_steps"]),
        "elapsed_seconds": float(elapsed),
    }
    return result
