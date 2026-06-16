# d=0.85 Failure Mode Diagnostic

Rule: `low_tv_spectral_teacher`

Target references: `ref027`, `ref033`, `ref049`

Radius: `0.85`

Split gate: `0.004`

## Conclusion

summary-level evidence favors multi-sector / heterogeneous-sector behavior over pure Monte Carlo noise.

Important limitation: No retained per-particle direction/sample arrays were found for these units. Therefore the projection and histograms below are replicate-summary projections/histograms, not raw shell-sample projections. A raw geometric sector test would require rerunning the same units with particle/projection retention enabled.

## Target Summary

| ref | overall split/P | rep 4-split range/P | half 4-split range/P | CE range | feature silhouette | status |
| --- | --- | --- | --- | --- | --- | --- |
| ref_027 | 0.005224 | 0.007035 | 0.007035 | 0.047042 | 0.257 | multi_sector_suspect |
| ref_033 | 0.005032 | 0.007048 | 0.007048 | 0.037150 | 0.277 | multi_sector_suspect |
| ref_049 | 0.005460 | 0.005942 | 0.005942 | 0.051545 | 0.439 | multi_sector_suspect |

## How This Was Tested From Existing Artifacts

- 4-split test 1: split the 16 retained replicate summaries into four consecutive replicate blocks and compare block `logmeanexp(logZ)` per parameter.
- 4-split test 2: split the 32 retained per-replicate half-logZ summaries into four consecutive half blocks and compare block `logmeanexp(logZ)` per parameter.
- CE test: inspect replicate-level weighted CE range and correlation with replicate logZ.
- Projection test: PCA projection of retained replicate-level summary features: `logZ/P`, weighted CE, weighted error, weighted H, ESS fraction, and internal split diff.
- Histogram test: replicate-summary histograms for centered logZ/P, weighted CE, and replicate internal split/P.

## Outputs

- `ref_diagnostics_summary.csv`
- `target_replicate_diagnostics.csv`
- `target_4split_blocks.csv`
- `artifact_inventory.csv`
- `replicate_feature_projection.csv`
- `figures/fig01_target_refs_4split_logZ_blocks.png`
- `figures/fig02_all_refs_4split_range_rank.png`
- `figures/fig03_target_refs_CE_vs_logZ.png`
- `figures/fig04_replicate_feature_projection.png`
- `figures/fig05_target_refs_logZ_CE_split_histograms.png`

## Artifact Inventory

Loaded d=0.85 unit summaries: `42`

Retained raw particle/sample arrays found: `0`
