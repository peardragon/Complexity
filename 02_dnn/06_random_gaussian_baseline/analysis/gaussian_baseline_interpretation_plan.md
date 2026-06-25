# Gaussian Random Baseline Interpretation Plan

This note fixes the interpretation rule for the Gaussian-random baseline run before
the final `phi(d)_energy` overlay is available.

## Current Complexity Evidence

The Gaussian-random synthetic dataset has already been generated under
`06_random_gaussian_baseline`, and its roughness-style complexity diagnostic has
been measured.

- Complexity proxy: kNN graph label roughness on normalized input features.
- Gaussian mean kNN edge disagreement: `0.501386`.
- Nearest spin-dynamics beta by this proxy: `beta=0.05`.
- Spin `beta=0.05` kNN edge disagreement mean: `0.478021`.
- Absolute gap to Gaussian: `0.023366`.

This is consistent with the Gaussian-random labels behaving like a high-temperature
or low-beta null baseline under this specific input-local label-roughness proxy.

## Theoretical Reading

The statement "similar complexity should imply similar `phi(d)`" is valid only
under a strong condition: the chosen scalar complexity measure must be close to a
sufficient statistic for the local energy/loss landscape probed by `phi(d)`.

That condition is plausible enough to test, but it is not guaranteed. The kNN
roughness proxy measures local label disorder in input space. The measured
`phi(d)` depends on additional structure:

- the trained/reference model distribution,
- the parameter-space distance shell,
- cross-entropy curvature near reference solutions,
- the model's inductive bias and optimization path,
- whether the synthetic Gaussian labels create the same kind of local decision
  boundary geometry as the spin-dynamics labels.

Therefore the random Gaussian baseline is a useful null test, not an automatic
theorem.

## Post-Overlay Decision Rule

Use the final overlay and Gaussian curve CSV to distinguish the following cases.

1. Similar complexity and similar `phi(d)`:
   This supports the claim that the proposed complexity measure captures the
   relevant axis controlling the local energy landscape. It is supporting
   evidence, not a proof, because two systems can agree on one scalar and one
   curve for accidental or limited-range reasons.

2. Similar complexity but different `phi(d)`:
   This is the most informative failure mode. It would mean the scalar roughness
   measure is not sufficient by itself. It may still be a valid complexity
   component, but it would not fully represent the kind of complexity that
   governs `phi(d)_energy`.

3. Different complexity and different `phi(d)`:
   This is consistent but weak evidence. It shows the null differs, but does not
   isolate whether the complexity proxy is the causal axis.

4. Different complexity but similar `phi(d)`:
   This would suggest `phi(d)` is insensitive to this complexity proxy over the
   tested range, or that other geometric factors dominate.

## Required Final Evidence

The goal should be considered complete only when all of these are true:

- `90` Gaussian synthetic datasets exist.
- Complexity diagnostics exist and identify the nearest spin beta.
- Reference search has `90` valid dataset units with `30` references each.
- Sampling manifest has `2700` rows.
- Dense shell sampling completes `675000` units with zero failures.
- Gaussian high-beta curve CSV exists.
- `phi(d)_energy` overlay figure exists under
  `06_random_gaussian_baseline/figures/gaussian_overlay`.
- The final verifier passes:

```bash
/home/bjyong/miniconda3/bin/python 06_random_gaussian_baseline/scripts/verify_gaussian_baseline_outputs.py
```

