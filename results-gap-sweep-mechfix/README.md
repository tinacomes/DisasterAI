# M2 cognitive-gap sweep (revised model) — the optimum is ROBUST to the cognitive profile, not dependent on it

Full 132-cell grid — gap g ∈ {0, 0.5, 1.0, 1.5} × D-midpoint ∈ {2, 3, 4} ×
11 α — main-model configuration, N=20 seed-paired replications, 200 ticks,
revised model (`60c96e2` = `30f89e0` mechanisms, linear dose).

- **Run**: [32866377905](https://github.com/tinacomes/DisasterAI/actions/runs/32866377905),
  2026-08-25, workflow *Run Gap Sweep only*. All 132 cells succeeded.
- `gap-cell-g{g}-dm{dm}-a{α}/bubble_gap_*.json` per cell;
  `gap-sweep-figure/gap_sweep.png` (CI collect over all cells).

## Verdict

1. **The interior optimum holds in every cell of the grid.** The
   operational optimum (unmet-needs minimum) is α\* = 0.6–0.7 in all 12
   (g, d_mid) cells; the population bubble composite
   (|SECI-pop| + |AECI-IE-chan-pop|) gives α\* = 0.6–0.8 in all 12. The
   dose–response itself is profile-invariant: explorer MAE runs
   ≈0.55 → ≈1.75 in every cell.
2. **The originally intended Fig. 3b claim — "the Goldilocks location
   depends on the population's cognitive profile" — is NOT supported at
   the population/operational level** (the surface is flat). The only
   gradient is in the per-type channel composite, whose α\* drifts upward
   with the gap (g=0: 0.2–0.5 → g=1.5: 0.6–0.7); treat as suggestive
   until M4 puts CIs on it.
3. **Reframe Fig. 3b as a robustness result**: the interior optimum is a
   structural property of the information ecosystem, not a knife-edge of
   the assumed D/δ agent mix — a direct pre-emption of the "your
   cognitive-parameter choices drive everything" review. This is the
   recommended paper framing (`PNAS_WRITING_INSTRUCTIONS.md` updated).
4. Comparison with the pre-revision sweep (`results-gap-sweep-fixed/`,
   step dose): there the population composite picked the α=0 corner in
   every complete cell — the same dose artifact as in the primary sweep;
   under the linear dose the interior optimum appears uniformly.

## Provenance note

The first archive pass captured only 99/132 cells:
`actions/download-artifact@v4`'s run-id listing truncates at 100
artifacts. `archive-artifacts.yml` now downloads via the REST API with
pagination (commit `1fe1f83`); the full grid is present.
