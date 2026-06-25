# Region Protocol

Regions are fixed before sampling:

1. `solution_core`: local low-loss neighborhood around the reference solution.
2. `near_same_valley`: still nearby, reachable only through a narrow valley.
3. `across_barrier`: low-loss mass behind a high ridge.
4. `remote_needle`: a thin distant low-loss basin.

The reference target mass is computed by dense-grid quadrature. Sampler failure
is judged against regions whose reference mass exceeds
`qc.min_truth_region_mass`.

