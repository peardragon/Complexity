# Dataset And Proxy Landscape

The proxy uses two collective coordinates around a reference solution. Low
energy basins represent solution-like regions; fixed ridges and minibatch
roughness terms represent dataset-induced barriers and local irregularity.

The target density is

```text
pi(z) proportional to exp(-beta * E(z))
```

where `E(z)` is a smooth soft-minimum over low-loss basins plus ridge and rough
dataset terms. The same energy is used by all methods.

The L2-common configs use

```text
E_target(z) = E_proxy(z) + lambda * ||z||^2
```

for every method. This keeps the target fixed across MC, HMC, pL, and vMF+L2;
only the sampling or proposal mechanism changes.

Configurations live in `config/`.
