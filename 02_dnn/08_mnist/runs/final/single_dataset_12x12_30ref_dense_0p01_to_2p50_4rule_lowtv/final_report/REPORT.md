# MNIST14 Single-Dataset 12x12 PM-SAIS Final Report

Architecture: `196-12-12-1-tanh`, P=2533

Scale: one split, 4 label rules, 30 exact references per rule, dense d_raw 0.01..2.50.

Main estimator: PM-SAIS averaged over references. `phi_by_rule_radius.csv` contains only QC-passed claim rows. `phi_by_rule_radius_raw_diagnostic.csv` contains dense raw diagnostic rows and must not be used as a claim table where QC failed.

Sampling rows: 30000 unit summaries. Sample-count mix: {256: 9881, 1024: 20119}.

## Complexity

| rule | TV | NMSTV |
| --- | --- | --- |
| low_tv_spectral_teacher | 0.146513 | 0.293026 |
| real_even_odd | 0.183485 | 0.366971 |
| teacher_nn | 0.306200 | 0.612400 |
| random_label | 0.507084 | 1.014168 |

## Sampling QC Coverage

| rule | QC-pass radii | total radii | pass radius range |
| --- | --- | --- | --- |
| low_tv_spectral_teacher | 16 | 250 | 0.01..0.17 |
| real_even_odd | 9 | 250 | 0.01..0.11 |
| teacher_nn | 0 | 250 | none |
| random_label | 2 | 250 | 0.01..0.02 |

Common supported radii across all rules: 0 / 250.

## Phi Outputs

| scope | file | rows | claim status |
| --- | --- | --- | --- |
| raw dense diagnostic | phi_by_rule_radius_raw_diagnostic.csv | 1000 | no-claim where QC failed |
| QC-pass rule/radius | phi_by_rule_radius.csv | 27 | claimable per passing rule/radius |
| common-radius claim table | final_claim_table.csv | 250 | 0 supported, 250 no_claim |

## Figures

| figure set | path | content |
| --- | --- | --- |
| label representatives | ../01_dataset_prepare/figures/label_representatives/ | 4 rule figures |
| raw dense energy | figures/fig08_phi_energy_raw_dense_diagnostic.png | all rules, 250 radii each |
| raw dense full | figures/fig09_phi_full_raw_dense_diagnostic.png | all rules, 250 radii each |
| QC-pass heatmap | figures/fig07_sampling_qc_pass_heatmap.png | pass/no-claim by rule and radius |
| QC-pass energy | figures/fig04_phi_energy_qc_pass_main.png | only passing rule/radius rows |

## Pipeline Comparison To Retained 3NN Synthetic

| item | retained 3NN synthetic | MNIST14 run |
| --- | --- | --- |
| model | 2-48-48-1 tanh, P=2545 | 196-12-12-1 tanh, P=2533 |
| input/data | synthetic 2D beta-cell datasets | one fixed MNIST 1/4 marginal, 12x12 inputs |
| label axis | many beta-indexed synthetic datasets | same input marginal, four label rules |
| complexity measure | dataset/rule complexity used for ordering | graph TV/NMSTV on the fixed MNIST marginal |
| reference ensemble | 30 references per retained beta/dataset setting | 30 exact optimizer-induced references per rule |
| shell sampler | hard L2 shell PM-SAIS with adaptive CE-tempered SMC | same hard L2 shell PM-SAIS estimator |
| phi aggregation | averaged over retained sample/reference units by beta/radius | averaged over 30 references for each rule/radius |
| QC claim policy | retained dense production claim where sampler QC passed | raw dense diagnostic exists, but common-radius claim is empty because teacher_nn fails split-QC |

## Interpretation

The dense raw phi(d) curves were obtained for all four rules over 250 radii. The formal claim table has no common supported radius because `teacher_nn` has zero QC-passed radii under the current 30-reference, mixed 1024/256-sample recovery run. The raw dense figures are therefore diagnostic curves, while the QC-pass files are the claimable subset.
