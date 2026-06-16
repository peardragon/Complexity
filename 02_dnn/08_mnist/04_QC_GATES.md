# 04_QC_GATES

## Global QC

- No NaN/Inf in datasets, references, shell summaries, or final aggregation.
- All random seeds must be recorded.
- Failed radius policy: `no_claim`.
- Failed stage policy: write `STAGE_BLOCKED.md`.

## Dataset QC

- `X_train` shape: `(n_train, 196)`.
- `X_test` shape: `(n_test, 196)`.
- labels are int values in `{-1,+1}`.
- train class balance by rule must be between 0.45 and 0.55 for smoke unless unavoidable ties are explained.
- `teacher_nn` threshold is train median logit.
- montage figure exists and visibly preserves digit shapes.

## Complexity QC

- kNN graph edge count > 0 for all k.
- all TV/NMSTV values finite.
- random-label NMSTV should be near random baseline after normalization; warn if not.
- do not fail solely because teacher/real ordering is unexpected.

## Reference QC

- selected references have `train_error == 0`.
- theta vector length equals P.
- pairwise duplicate threshold: selected theta L2 distance > 1e-6.
- per dataset smoke target: 5 exact references.
- if random_label exact success is insufficient, first increase attempts; only then consider backup architecture.

## PM-SAIS QC

Smoke thresholds:

| field | threshold |
|---|---:|
| finite unit fraction per rule/radius | >= 0.90 |
| q05 ESS fraction | >= 0.02 |
| max split logZ/P diff | <= 0.004 |
| bootstrap sd phi | <= 0.012 |
| CESS if SMC used | >= 0.60 |

Candidate/final thresholds:

| field | threshold |
|---|---:|
| finite unit fraction per rule/radius | >= 0.95 |
| q05 ESS fraction | >= 0.04 |
| max split logZ/P diff | <= 0.004 |
| bootstrap sd phi | <= 0.012 |

## Final claim QC

A radius is claimable only if all three rules pass QC at that radius and the reference count is adequate.

Report supported range explicitly:

```text
Supported d_raw radii: [...]
No-claim d_raw radii: [...]
```
