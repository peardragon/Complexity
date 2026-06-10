# Shell-Pool Parallel Contract

This document records the cross-stage contract between the promoted
`01_theory/02_theory_sampling` shell pool and the promoted
`02_dnn/04_sampling` shell pools.

## Contract

The theory and DNN shell-pool pipelines are parallel at the methodological
level:

- Both estimate local shell free energy around retained reference points.
- Both use an L2 shell parameterization
  `theta = theta_ref + sqrt(dimension) * radius * direction`.
- Both use vMF shell proposals centered on the retained reference direction.
- Both weight shell particles by the system loss or cross-entropy term.
- Both report log-partition estimates through log-mean-exp style estimators.
- Both aggregate per-reference/per-radius estimates into compact summary tables
  that drive figures and downstream proxy summaries.

The two pipelines are not identical at the runtime-policy level:

- Theory uses direct vMF importance sampling when split/ESS checks pass, with
  adaptive CE-tempered SMC fallback.
- DNN retained finals use adaptive CE-tempered SMC for the promoted shell
  estimates.

The defensible claim is therefore: same shell-estimation framework and QC
contract, different physical/statistical system, with a documented runtime
policy difference.

## System Boundary

Theory system:

- stage: `01_theory/02_theory_sampling`
- system: two-pool perceptron at alpha = 0.1
- promoted shell pool:
  `01_theory/02_theory_sampling/raw_outputs/shell_pool/`

DNN production system:

- stage: `02_dnn/04_sampling`
- system: 3NN, architecture `2-48-48-1-tanh`, `P=2545`
- public identity: `18_beta_cell_30_dataset_30_reference`
- promoted shell raw path:
  `02_dnn/04_sampling/raw_outputs/shell_pool/18_beta_cell_30_dataset_30_reference/d_0.01_to_2.50_dense/`

DNN completed extension:

- public identity: `18_beta_cell_60_dataset_30_reference`
- retained shell summary path:
  `02_dnn/04_sampling/raw_outputs/shell_pool/18_beta_cell_60_dataset_30_reference/d_0.01_to_2.50_dense/`
- retained status: compact summary tables and figures; row-level shell payloads
  are not retained in this path.

## Evidence Snapshot

Theory promoted shell pool:

- retained final sample units: 16,800
- base particles: 2,048
- far-radius replacement particles: 32,768
- high-particle bad-split replacements: 262,144 for 15 replacement units

DNN production shell pool:

- selected beta cells: 18
- selected datasets per beta: 30
- selected references per dataset: 30
- shell radius range: `d_0.01_to_2.50_dense`
- raw sample payload path:
  `02_dnn/04_sampling/raw_outputs/shell_pool/18_beta_cell_30_dataset_30_reference/d_0.01_to_2.50_dense/`
- proxy summary path:
  `02_dnn/05_proxy_local_entropy/raw_outputs/18_beta_cell_30_dataset_30_reference/d_0.01_to_2.50_dense/`

DNN completed extension:

- selected beta cells: 18
- datasets per beta: 60
- selected references per dataset: 30
- sampling rows: 8,100,000
- failed units: 0
- proxy rows: 4,500 each for absolute, delta, and derivative tables

## Source and Provenance Pointers

Theory implementation:

- active wrapper: `01_theory/02_theory_sampling/src/cli.py`
- promoted run report:
  `01_theory/02_theory_sampling/raw_outputs/shell_pool/run_report.md`

DNN implementation:

- sampler core: `02_dnn/04_sampling/src/pm_sais_core.py`
- vMF utilities: `02_dnn/04_sampling/src/vmf.py`
- model/loss code: `02_dnn/04_sampling/src/dnn_model.py`
- proxy tables: `02_dnn/05_proxy_local_entropy/src/make_proxy_tables.py`

## Rollback

Any future alignment run should leave the current promoted outputs in place
until replacement outputs pass QC. Superseded outputs must be moved under
`99_backup/cleanup_<timestamp>/` with a manifest before removal from the active
tree.
