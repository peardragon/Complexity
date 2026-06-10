# Theory Sampling

This stage keeps the promoted two-pool perceptron sampling final used to check
the shell estimator against the analytic theory stage.

## Method

- reference pool: retained exact/valid perceptron reference solutions
- shell pool: L2 shell samples around each reference
- shell parameterization: `theta = theta_ref + sqrt(N) * radius * direction`
- proposal: vMF-centered shell proposal
- runtime policy: direct vMF importance sampling when QC passes, with adaptive
  CE-tempered SMC fallback

## Active Outputs

- `raw_outputs/reference_pool/`: promoted reference solutions
- `raw_outputs/shell_pool/`: promoted shell estimates and QC summaries
- `figures/`: sampling-only diagnostic figures

Shell-pool parity with the DNN sampling stage is documented in
`../../SHELL_POOL_PARALLEL_CONTRACT.md`.
