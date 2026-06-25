# Claim Evidence Matrix

| Claim | Evidence in this workspace | Guardrail |
| --- | --- | --- |
| The proxy dataset creates several meaningful low-loss regions. | Grid reference masses in `02_region_protocol/raw_outputs/<id>/region_reference.csv`. | Region mass must be non-negligible under the target distribution. |
| Local HMC, pseudo-Langevin, and random-walk MCMC miss at least one important region. | Region hit counts and mass estimates in `05_qc_and_figures/raw_outputs/<id>/region_mass_estimates.csv`. | Failure is finite-budget reachability failure, not formal impossibility. |
| The final vMF + L2 methodology recovers the missing regions. | Self-normalized importance estimates and ESS in `04_vmf_l2_importance/raw_outputs/<id>/importance_summary.json`. | This is treated as the proxy instantiation of the final method, not as an unrelated baseline. Must pass ESS and per-region hit QC. |
| The result is relevant to the DNN complexity project. | The landscape is a shell-like proxy with region directions, L2 radii, weighted CE-like energy, and the same QC language as `../../02_dnn/04_sampling`. | It remains a proxy until mapped to actual DNN checkpoints. |

## Reproducibility contract

For existing methods, every run must report:

- the paper source being proxied;
- the proxy mapping from the paper method to this landscape;
- all sampler hyperparameters;
- initialization, burn-in, thinning, and sample count;
- acceptance or diffusion diagnostics;
- region reachability and QC failure reason.

For the final vMF + L2 method, every run must report:

- the proposal components, region direction anchors, L2 shell centers, and broad component weight;
- the exact target density used in the denominator-free importance ratio;
- self-normalized region estimates;
- ESS fraction and per-region hit counts;
- the same QC thresholds used for the baselines.
