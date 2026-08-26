# M4 statistics pass — mixed-model regressions + Holm-corrected contrasts

Run locally (statsmodels 0.14.6) over the canonical dataset
`results-mechfix/` (2 configurations × 11 α × 20 seeds = 440 late-run
observations per outcome; commit `1999bfb` state of the archive; script
`tools/sweep_regression.py`, extended with the two population-level
outcomes). Outputs: `regression_table.md` (SI Table S6 basis:
standardized coefficients; SI Table S1 basis: Holm-corrected per-level
paired contrasts) and `regression_table.csv`.

## Headline statistical results

- **The interior optimum's curvature is significant everywhere it
  matters**: unmet needs α² = +9.24 (SE 0.71)*** with α = −7.50***
  (the U-shape), and the α²×configuration interaction −6.47*** (the U is
  shallower under bounded access — the buffering); population
  AECI-IE-chan α = +3.76***, α² = −3.26*** (the served-information
  U-shape); explorer precision α² = −7.96***.
- **Configuration main effects** (main − control, SD units): unmet needs
  −0.67***, MAE_exploit −0.55***, MAE_explor −0.14**, precision_exploit
  +0.57***, precision_explor +0.26***, population AECI-IE-chan −0.82***
  — the main model is significantly better on every operational outcome
  while carrying a significantly narrower served-information environment.
- **Per-level Holm contrasts** (N=20, full α grid) agree with the N=50
  boundary table: ΔSECI_exploit −0.290 [−0.386, −0.195], adjusted
  p ≈ 5e-05 at α=1 (N=50: −0.269, p ≈ 5e-08); mid-α SECI deltas are not
  significant, exactly as Finding 1 states (the configuration contrast is
  a high-α phenomenon). Explorer precision becomes significant from
  α=0.7 (+0.022) rising to +0.413*** at α=1.
- Method note for the SI: several outcomes fell back from MixedLM to
  seed-clustered OLS (flagged per row in the table) — with a
  random-intercept-only design and z-scored outcomes the two are
  near-equivalent; report the method column as-is.

For the paper: **Table S1** = the Holm contrast blocks here (α ≤ 0.7)
merged with `results-boundary-n50/boundary-deltas/` (α ≥ 0.8, N=50);
**Table S6** = the coefficients table.
