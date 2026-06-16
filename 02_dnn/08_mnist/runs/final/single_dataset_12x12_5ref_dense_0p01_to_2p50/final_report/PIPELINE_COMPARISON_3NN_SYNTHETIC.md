# Pipeline Comparison Against 3NN Synthetic Production

This document compares the completed MNIST14 single-dataset PM-SAIS run with
the retained 3NN synthetic production pipeline. The comparison is based on:

- `02_dnn/04_sampling/raw_outputs/shell_pool/18_beta_cell_30_dataset_30_reference/d_0.01_to_2.50_dense/run_config.json`
- `02_dnn/08_mnist/runs/final/single_dataset_12x12_5ref_dense_0p01_to_2p50/03_pool_design/run_config_resolved.json`
- `02_dnn/08_mnist/runs/final/single_dataset_12x12_5ref_dense_0p01_to_2p50/05_pool2_pm_sais_sampling/QC_STATUS.json`
- `02_dnn/08_mnist/runs/final/single_dataset_12x12_5ref_dense_0p01_to_2p50/final_report/QC_STATUS.json`

## Executive Summary

The core estimator is the same: hard L2 shell PM-SAIS with vMF proposal,
adaptive CE-tempered SMC, `lambda_reg=1.0`, `r0=0.01`, dense radii
`0.01..2.50`, and 1024 samples per reference-radius unit.

The major differences are the problem axis and statistical scale. The 3NN
production run estimates a beta-conditioned synthetic ensemble over retained
beta cells, many datasets, and many references. The MNIST14 run estimates a
single MNIST14 split over three label rules with five references per rule.
Therefore the MNIST14 run is methodologically aligned but has much weaker
production-level claim strength and a much narrower supported radius set.

## Parameter And Scope Comparison

| Item | 3NN synthetic production | MNIST14 completed run | Pipeline implication |
|---|---:|---:|---|
| Public identity | `18_beta_cell_30_dataset_30_reference` | `single_dataset_12x12_5ref_dense_0p01_to_2p50` | Different experimental object and public namespace. |
| Dataset family | Synthetic 2D classification datasets over beta cells | MNIST downsampled to 14x14, one train/test split | Dataset generator changed; PM-SAIS shell logic unchanged. |
| Input dimension | `2` | `196` | Network input layer changed only. |
| Network | `2-48-48-1-tanh` | `196-12-12-1-tanh` | Parameter count kept near production scale while changing domain. |
| Parameter count `P` | `2545` | `2533` | Comparable parameter scale; runtime is not from oversized P. |
| Ensemble axis | 18 retained beta cells x 30 datasets x 30 references | 1 split x 3 label rules x 5 references | MNIST14 is single-dataset/scaled, not production ensemble scale. |
| Label / condition axis | Beta cell / synthetic cell identity | `real_even_odd`, `teacher_nn`, `random_label` | Rule comparison replaces beta-conditioned phase map. |
| Reference pool | 30 exact refs per dataset | 5 exact refs per rule | Lower reference count increases uncertainty and no-claim risk. |
| Radius grid | `0.01..2.50`, step `0.01` | `0.01..2.50`, step `0.01` | Same dense distance scan. |
| Baseline radius | `r0=0.01` | `r0=0.01` | Same delta-phi anchor. |
| Samples per ref/radius | `1024` | `1024` | Same base unit sample budget. |
| Unit count | 18 x 30 x 30 x 250 = 4,050,000 shell units | 3 x 5 x 250 = 3,750 shell units | MNIST14 is about 1080x smaller in shell-unit count. |

## Estimator Comparison

| Estimator component | 3NN synthetic production | MNIST14 completed run | Same? |
|---|---|---|---|
| Sampling method | `exact_shell_l2_vmf_adaptive_ce_tempered_smc` | `exact_shell_l2_vmf_adaptive_ce_tempered_smc` | Yes |
| Shell definition | `theta = theta_ref + sqrt(P) * d * u` | `theta = theta_ref + sqrt(P) * d * u` | Yes |
| Distance convention | `d_raw = ||theta - theta_ref|| / sqrt(P)` | `d_raw = ||theta - theta_ref|| / sqrt(P)` | Yes |
| Proposal family | vMF shell proposal | vMF shell proposal | Yes |
| Tempering | Adaptive CE-tempered SMC | Adaptive CE-tempered SMC | Yes |
| L2 prior coefficient | `lambda_reg=1.0` | `lambda_reg=1.0` | Yes |
| Full log partition | `logZ_inf_stripped + reference_prior_log_weight` | `logZ_inf_stripped + reference_prior_log_weight` | Yes |
| Energy phi | Delta full-logZ energy divided by `P` | Delta full-logZ energy divided by `P` | Yes |
| Full phi | Energy phi plus L2 shell area term | Energy phi plus L2 shell area term | Yes |
| Derivative outputs | Production derivative tables are retained | Not generated in this MNIST14 run | No |
| H-threshold phase maps | Retained through proxy-local-entropy stage | Not generated in this MNIST14 run | No |

## Stage-Level Pipeline Comparison

| Pipeline stage | 3NN synthetic production | MNIST14 completed run | Difference |
|---|---|---|---|
| 00 contract/docs | Production retained layout across `01_dataset_gen` to `05_proxy_local_entropy` | Run-local contract under `02_dnn/08_mnist/runs/final/...` | MNIST14 avoids modifying old retained outputs. |
| 01 dataset | Synthetic raw datasets grouped by beta cell and dataset seed | One MNIST14 split with three label rules | Domain and conditioning axis changed. |
| 02 complexity | Synthetic cell/dataset complexity summaries | NMSTV/TV diagnostics for the three MNIST14 rules | Same role, different graph/data object. |
| 03 reference search | Exact optimizer-induced refs per synthetic dataset | Exact optimizer-induced refs per label rule | Same role, reduced reference scale. |
| 04/05 shell sampling | Production hard-shell PM-SAIS over 4,050,000 units | Hard-shell PM-SAIS over 3,750 units | Same estimator, much smaller unit grid. |
| Stability handling | Production config uses fixed SMC kernel | MNIST14 uses pilot-driven fallback policies for 50 units | Operational difference introduced to pass split QC safely. |
| Proxy phi tables | `absolute_phi_by_beta_radius.csv`, `delta_phi_by_beta_radius.csv`, derivative/HQ tables | `phi_by_rule_radius.csv`, `phi_bootstrap_by_rule_radius.csv`, `final_claim_table.csv` | MNIST14 produces rule-wise final phi only, not full proxy dashboard tables. |
| Figures/dashboard | Interactive local entropy dashboard and production phase-map figures | Static final phi and QC figures | Output surface is smaller and run-local. |
| Claim policy | Production-scale beta/radius retained outputs | Only QC-passed common radii are claimable | MNIST14 final claim is narrow. |

## QC And Claim Comparison

| QC item | 3NN synthetic production config | MNIST14 completed run | Interpretation |
|---|---:|---:|---|
| `max_split_logZ_per_P_diff` | `0.004` | `0.004` | Same split-logZ tolerance. |
| `bootstrap_sd_phi_max` | `0.012` | `0.012` | Same phi bootstrap tolerance. |
| `q05_ess_fraction_min` | `0.2` | `0.04` | MNIST14 uses a looser ESS gate. |
| Stage 05 unit rows | Production-scale shell grid | `3750` | MNIST14 completed all planned units. |
| Stage 05 hard-shell max abs error | Hard-shell exactness expected | `8.881784197001252e-16` | Hard-shell geometry passed. |
| Stage 05 fallback units | Not part of fixed production config | `50` | Fallback policies were needed for unstable pilot cases. |
| Stage 05 no-claim rows | Production claim table not re-audited here | `716 / 750` rule/radius rows | Most MNIST14 rule/radius rows are not claimable. |
| Common supported radii | Dense production range retained | `0.01, 0.02, 0.04, 0.05` | Final MNIST14 phi curve is supported only at four radii across all rules. |
| Final phi rows | Beta/radius tables | `34` rule/radius rows | Only QC-passed rule/radius rows are plotted/exported. |
| Final figures | Dashboard/phase-map production outputs | 3 final static figures | Final output is intentionally compact. |

## Bottom Line

The MNIST14 run did obtain a final `phi(d)` output, but the result should be
read as a scaled single-dataset methodology transfer, not as a production-scale
replacement for the 3NN synthetic ensemble. The estimator mechanics are aligned
with 3NN production; the domain, ensemble axis, reference count, fallback
operation, and downstream proxy/dashboard breadth are the main differences.
