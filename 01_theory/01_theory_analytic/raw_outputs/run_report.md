# Analytic Theory Raw Output Report

## Config

- Config file: `01_theory/01_theory_analytic/config/default.json`
- Alpha: `0.1`
- Lambda ref/shell: `1` / `1`
- Radius grid: 42 points from `0.15` to `2.20` in steps of `0.05`
- Baseline grid: `q_grid_count=45`, `s_grid_count=31`, `s_abs_max=0.25`

## Eq. (50) Normalization

The regenerated outputs use the corrected selected-reference energetic
quadrature: the conditional reference average at fixed `z0` uses the `z1`
Gaussian weight and denominator `H0(z0)`, and the outer `Dz0` quadrature weight
is applied exactly once.

Focused normalization checks from the regenerated source:

- Constant-integrand Eq. (50) quadrature normalization: `1.0016879917552934`
- Legacy duplicated-`z0` normalization on the same check: `0.09896283472721967`
- Baseline/full-feasible `A=0` energetic-term absolute difference at a fixed
  test point: `5.551115123125783e-17`

## Runtime

- Python: `D:\Complexity\.venv\Scripts\python.exe`
- Baseline regeneration wall time: `6.010 s`
- Error log: `theory_full_rs_alpha0p1.err.log`, size `0`

## Output Files

- `theory_full_rs_alpha0p1.csv`: corrected Eq. (50) analytic full-RS baseline
  table with 42 radius rows. The `phi_rel` column is relative to the first
  radius and is used by the analytic figure and dense comparison stage.
- `theory_full_rs_alpha0p1.log`: stdout from the regeneration command.
- `theory_full_rs_alpha0p1.err.log`: stderr from the regeneration command.

## Reproduction Command

```powershell
.\.venv\Scripts\python.exe 01_theory\01_theory_analytic\src\theory_full_rs.py `
  --config 01_theory\01_theory_analytic\config\default.json `
  --out 01_theory\01_theory_analytic\raw_outputs\theory_full_rs_alpha0p1.csv
```
