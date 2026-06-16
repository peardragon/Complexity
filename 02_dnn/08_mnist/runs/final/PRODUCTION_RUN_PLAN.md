# MNIST14 Final Production Run Plan

## Options and Selected Path

- Direct PM-SAIS was rejected for final because smoke random-label ESS collapsed beyond baseline.
- Final uses the retained 02_dnn/04_sampling production pattern: exact-shell L2 vMF proposal plus adaptive CE-tempered SMC.

## Final Scale

- splits: 10
- train size: 1024
- refs per dataset/rule: 20
- dense radii: 0.01..2.50 step 0.01 (250 radii)
- shell units: 150000
- particles per unit: 1024

## Risks

- Random-label exact references may fail under 196-16-16-1; if so, increase attempts first, then evaluate the documented 24-24 backup architecture explicitly.
- Full sampling may be day-scale; per-unit summaries make the run resumable.
- Failed radii stay no_claim and are excluded from final phi curves.

## Rollback

All final artifacts are isolated under `02_dnn/08_mnist/runs/final`; old retained 02_dnn/05 outputs are read-only references and are not modified.
