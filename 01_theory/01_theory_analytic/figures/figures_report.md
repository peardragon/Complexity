# Analytic Theory Figures Report

This folder retains the active analytic-stage figure generated from the
corrected Eq. (50) theory CSV.

## Input Data

- `01_theory/01_theory_analytic/raw_outputs/theory_full_rs_alpha0p1.csv`

The CSV contains the alpha `0.1` full-RS analytic baseline over the active
42-point radius grid, with columns for `r`, `phi`, `phi_rel`, `Q`, `p`, `t`,
`cd`, `s`, `qref`, and `alpha`.

## Figure

- `fig01_phi_by_analytic_solution_alpha0p1.png`: analytic full-RS
  `phi(d)-phi(d0)` curve generated directly from the corrected baseline CSV.
  It does not include sampling finite-N curves; those are shown in the
  comparison-stage figure.
