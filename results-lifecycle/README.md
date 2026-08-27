# Per-seed lifecycle re-run of the primary sweep — CI-grade dynamics grid

Instrumented re-run of the full primary sweep (11 α × 2 configurations ×
N=20 seed-paired replications × 200 ticks) with the per-seed lifecycle
columns (`lc_*_runs`) emitted by the instrumented `_aggregate`
(commit `909596d`): chamber formation tick, peak depth, episode count,
final dissolution (right-censored), persistence, end-in-chamber flag —
per type and population — plus per-type capture onset. Thresholds as in
`tools/lifecycle_metrics.py` / `test_filter_bubbles.py`.

- **Run**: [33063104901](https://github.com/tinacomes/DisasterAI/actions/runs/33063104901),
  2026-08-27, workflow *Compare Baseline vs Network/Mobility Switches*,
  dispatched on branch head `1f2e7c4`.
- Derived per-seed tables: [`lifecycle/lifecycle_perseed.md`](lifecycle/lifecycle_perseed.md)
  (+ the mean-trajectory tables/figures of the prototype layer,
  regenerated from this dataset).

## Status relative to the canonical archive — read this first

This dataset is **NOT bit-identical** to `results-mechfix/` (run
32821105202, commit `30f89e0`): intervening code changes (e.g. the
epsilon-decay instrumentation) shifted RNG draw order, so per-seed
values drift while **all aggregates agree within seed noise** (checked
column-by-column; e.g. explorer MAE 0.558 vs 0.540 at α=0, 1.738 vs
1.741 at α=1; L1+ pool 28.1 vs 28.6 at α=1). This is the same
documented behaviour as the earlier `results-final` reproduction.
**Main-text endpoint numbers continue to cite `results-mechfix/`**;
this archive is citable for the per-seed lifecycle statistics, which
only it carries.

One replication-sensitivity note: the composite α\* table of this run
puts 5/12 definitions at 0.6 (spread 0.0–0.9) versus the canonical
7/12 (spread 0.1–0.6) — the **population composites remain at 0.6**,
consistent with the compendium's finding that they are the most robust
α\* definition. The paper's 7/12 claim is a property of the canonical
dataset and should stay attributed to it; if desired, this replication
belongs in the Table S2/S5 transparency material.

## Headline per-seed readings (CI-grade, full grid)

1. **Formation is universal for the accuracy-seekers**: 20/20
   replications form a chamber in every (α, configuration) cell;
   formation is delayed by starvation at high α (tick 22 ± 1 at α=0 →
   67 ± 7 at α=1, main model) with peak depths in a narrow band.
2. **Dissolution collapses with dose**: dissolved by the 200-tick
   horizon — main model 17/20 (α=0) → 15/20 (α=0.6) → 3/20 (α=1);
   control 20/20 → 19/20 → 3/20. Matches the M9 anchors
   (`results-reversal/`) cell-for-cell where they overlap.
3. **The Finding-1 contrast, per seed**: at α=1 the
   confirmation-seekers' chamber is standing at the end in 16/20
   (bounded) vs 2/20 (unrestricted) replications; in the control it
   dissolves for good in 20/20 at α=1.
4. **Capture onset accelerates ~25×**: exploiter AI-majority onset
   59 ± 3 (α=0) → 13 ± 4 (α=0.7) → 2 ± 1 (α≥0.9), reached in
   19–20/20 replications at every level (main model).
5. **The societal lens misleads dynamically too**: the population-level
   chamber is standing at the end in 19/20 main-model runs at α=1 but
   only 1/20 control runs — the per-type fragmentation view of the
   main text is the informative one.

Regeneration: `python3 tools/lifecycle_metrics.py --switches
plots-config-switches/experiment_results.json --baseline
plots-config-baseline/experiment_results.json --outdir lifecycle`.
