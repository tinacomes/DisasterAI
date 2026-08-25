# M3 boundary strengthening (N=50, high α) — Finding 1 and the buffering claims are seed-robust: PASS

High-alignment boundary α ∈ {0.8, 0.9, 1.0} in BOTH configurations,
**N=50** seed-paired replications (seeds 0–49, superset of the primary
sweep's 0–19), 200 ticks, revised model (`2836a18` = `30f89e0`
mechanisms). Paired per-seed deltas (main − control) with Holm-corrected
p-values: `boundary-deltas/boundary_deltas.md` (+ .csv/.txt).

- **Run**: [32894013398](https://github.com/tinacomes/DisasterAI/actions/runs/32894013398),
  2026-08-25, workflow *Run Boundary Sweep (M3)*. All 6 cells succeeded.

## Verdict — every headline boundary claim survives at N=50 with Holm correction

- **Structural precondition (Finding 1)**: ΔSECI_exploit (main − control)
  = −0.115 / −0.169 / **−0.269** at α = 0.8/0.9/1.0, all adjusted
  p ≤ 0.003 — network-bounded access keeps the confirmation-seekers'
  chamber significantly deeper than unrestricted access at every boundary
  level. The explorer deltas are small (−0.08/−0.10, significant at
  0.8/0.9; n.s. at 1.0), consistent with the explorer chamber deepening
  in *both* configurations.
- **Operational buffering (Finding 3)**: Δunmet = −0.78 / −2.21 /
  **−6.56** (p ≤ 2.6e-07 throughout); Δprecision_explor +0.07 → **+0.33**
  and Δprecision_exploit +0.13 → +0.15; ΔMAE negative for both types at
  all three α — the main model dominates the control on every operational
  outcome at the boundary.
- **AI-channel contrast (new, per-type)**: under bounded access the AI
  serves exploiters a significantly narrower pool
  (ΔAECI-IE-chan_exploit −0.341 at α=1, p ≈ 3e-14) but explorers a
  significantly MORE diverse one (+0.229, p ≈ 2e-06) — the configuration
  reshapes the AI channel in opposite directions for the two cognitive
  styles. Worth one sentence in Finding 1 or the SI.

## Use in the paper

These are the Table S1 numbers for the α ≥ 0.8 rows (M4 extends the same
machinery to the full α grid). Cite deltas from `boundary_deltas.md`
verbatim; per-cell series live in `boundary-{config}-{α}/`.
