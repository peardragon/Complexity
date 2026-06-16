# MNIST14 Single-Dataset 12x12 PM-SAIS Final Report

Architecture: `196-12-12-1-tanh`, P=2533

Scale: one split, 4 label rules, 60 exact references per rule, sparse-wide hard d_raw grid from 0.01 to 2.50.

Main estimator: PM-SAIS averaged over references. `phi_by_rule_radius.csv` contains only QC-passed claim rows. `phi_by_rule_radius_raw_diagnostic.csv` contains sparse-wide raw diagnostic rows and must not be used as a claim table where QC failed.

Sampling rows: 4560 unit summaries. Sample-count mix: {2048: 4560}.

## Complexity

| rule | TV | NMSTV |
| --- | --- | --- |
| low_tv_spectral_teacher | 0.189503 | 0.379006 |
| real_even_odd | 0.253405 | 0.506810 |
| teacher_nn | 0.343267 | 0.686535 |
| random_label | 0.491038 | 0.982076 |

## Sampling QC Coverage

| rule | QC-pass radii | total radii | pass radius range |
| --- | --- | --- | --- |
| low_tv_spectral_teacher | 8 | 19 | 0.01..0.20 |
| real_even_odd | 1 | 19 | 0.01..0.01 |
| teacher_nn | 5 | 19 | 0.01..0.08 |
| random_label | 4 | 19 | 0.01..0.08 |

Common supported radii across all rules: 1 / 19.

## Phi Outputs

| scope | file | rows | claim status |
| --- | --- | --- | --- |
| raw sparse-wide diagnostic | phi_by_rule_radius_raw_diagnostic.csv | 76 | no-claim where QC failed |
| QC-pass rule/radius | phi_by_rule_radius.csv | 18 | claimable per passing rule/radius |
| common-radius claim table | final_claim_table.csv | 19 | 1 supported, 18 no_claim |

## Figures

| figure set | path | content |
| --- | --- | --- |
| label representatives | ../01_dataset_prepare/figures/label_representatives/ | 4 rule figures |
| raw sparse-wide energy | figures/fig08_phi_energy_raw_dense_diagnostic.png | all rules, 19 radii each |
| raw sparse-wide full | figures/fig09_phi_full_raw_dense_diagnostic.png | all rules, 19 radii each |
| QC-pass heatmap | figures/fig07_sampling_qc_pass_heatmap.png | pass/no-claim by rule and radius |
| QC-pass energy | figures/fig04_phi_energy_qc_pass_main.png | only passing rule/radius rows |

## Pipeline Comparison To Retained 3NN Synthetic

| item | retained 3NN synthetic | MNIST14 run |
| --- | --- | --- |
| model | 2-48-48-1 tanh, P=2545 | 196-12-12-1 tanh, P=2533 |
| input/data | synthetic 2D beta-cell datasets | one fixed MNIST 1/4 marginal, 12x12 inputs |
| label axis | many beta-indexed synthetic datasets | same input marginal, four label rules |
| complexity measure | dataset/rule complexity used for ordering | graph TV/NMSTV on the fixed MNIST marginal |
| reference ensemble | 30 references per retained beta/dataset setting | 60 exact optimizer-induced references per rule |
| shell sampler | hard L2 shell PM-SAIS with adaptive CE-tempered SMC | same hard L2 shell PM-SAIS estimator |
| phi aggregation | averaged over retained sample/reference units by beta/radius | averaged over 60 references for each rule/radius |
| QC claim policy | retained dense production claim where sampler QC passed | sparse-wide diagnostic plus QC-pass claim subset |

## Interpretation

The sparse-wide raw phi(d) curves were obtained for all four rules over 19 hard radii. The formal claim table should be interpreted through the QC-pass subset; raw sparse-wide figures are diagnostic wherever a rule/radius failed QC.
