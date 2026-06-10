# DNN Sampling

This stage estimates local shell statistics around trained 3NN reference
solutions. It is the DNN analogue of the theory shell-sampling stage, with an
all-SMC runtime policy for retained DNN finals.

## Method

- architecture: `2-48-48-1-tanh`, `P=2545`
- shell parameterization:
  `theta = theta_ref + sqrt(P) * radius * direction`
- sampler core: `src/pm_sais_core.py`
- proposal utilities: `src/vmf.py`
- model/loss evaluation: `src/dnn_model.py`
- retained DNN sampler policy:
  `exact_shell_l2_vmf_adaptive_ce_tempered_smc`
- QC quantities: ESS fractions, split logZ/P differences, SMC CESS,
  acceptance, weighted accuracy, CE/L2 ratios, and failed-unit manifests

## Active Outputs

Production run:

- reference subset:
  `raw_outputs/reference_pool/18_beta_cell_30_dataset_30_reference/`
- shell raw payloads:
  `raw_outputs/shell_pool/18_beta_cell_30_dataset_30_reference/d_0.01_to_2.50_dense/`

Completed extension:

- reference pool:
  `raw_outputs/reference_pool/18_beta_cell_60_dataset_30_reference/`
- shell summary:
  `raw_outputs/shell_pool/18_beta_cell_60_dataset_30_reference/d_0.01_to_2.50_dense/`

The 18-beta/30-dataset reference subset points back to the retained 36-beta
source reference pool to avoid duplicating theta payloads.

Shell-pool parity with the theory sampling stage is documented in
`../../SHELL_POOL_PARALLEL_CONTRACT.md`.
