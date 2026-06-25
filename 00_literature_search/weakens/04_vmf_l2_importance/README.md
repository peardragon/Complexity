# Final Method: vMF Direction Plus L2 Radius Importance Sampling

This directory represents the method we intend to use, specialized to the
proxy landscape. It should not be interpreted as a separate auxiliary baseline.

The final-method proxy uses the landscape figure itself:

1. Convert each pre-registered region center into a direction and radius around
   the reference solution.
2. Sample directions using a vMF distribution on the unit circle. In two
   dimensions this is the von Mises distribution.
3. Sample L2 radii with truncated normal shells centered at the region radii.
4. Include a broad uniform safety component.
5. Reweight samples by `exp(-beta * E(z)) / q(z)`.

The estimator is the same importance-sampling object used in the DNN shell
setting:

```text
target(z) proportional to exp(-beta E(z))
w_i = target_unnormalized(z_i) / q(z_i)
E_pi[f] approx sum_i w_i f(z_i) / sum_i w_i
```

The proxy proposal factorizes into a vMF direction term and an L2 radius term.
In 2D the vMF direction term is the von Mises distribution on the unit circle.
In the DNN setting the same role is played by vMF directions on the
high-dimensional L2 shell.

QC uses self-normalized region mass, ESS fraction, and per-region hit counts.
