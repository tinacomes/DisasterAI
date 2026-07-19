# Committed Results

All results in this directory come from one seed-paired GitHub Actions run of the
**Compare Baseline vs Network/Mobility Switches** workflow:

- **Run**: [29100134858](https://github.com/tinacomes/DisasterAI/actions/runs/29100134858), 2026-07-10
- **Code version**: commit `3317fda` (post-fix code; C1–C12 review fixes applied)
- **Parameters**: 11 α levels (0.0–1.0), 20 replications per α, 200 ticks per run.
  Replicate *i* uses seed *i* in both configurations, so cross-configuration
  deltas are paired per seed.

## Directories

| Directory | Contents | Configuration |
|---|---|---|
| `switched-sweep/` | Figures, `summary_table.csv/md`, `experiment_results.json` | mobility = 1, `network_type = spatial_bridged`, `query_scope = network` |
| `baseline-sweep/` | Same file set | mobility = 0, `network_type = components`, `query_scope = global` |
| `comparison/` | `comparison_table.csv/md` (raw metrics with paired per-seed deltas), `comparison_configs.png` overlay figure, `comparison.txt` console log | baseline vs switched |

`experiment_results.json` in each sweep directory contains the full per-α
aggregated metrics and supports figure regeneration without re-simulation
(`test_filter_bubbles.py --plots-only`, or the **Replot Primary Sweep** workflow).

## Interpretation reminders

- Signed echo indices (SECI, AECI-Var, AECI-Err): **negative = echo chamber**.
- Normalised composites in `summary_table.*` are range-normalised *within one
  sweep*; never compare them across configurations. Cross-configuration claims
  use the raw-metric paired deltas in `comparison/comparison_table.md`.
- Spatial coverage maps average 20 randomised epicentres and are not
  interpretable as maps; use the Spatial Visualization workflow
  (fixed epicentre) for map figures.

## Regenerating or updating

To reproduce from scratch, dispatch **Compare Baseline vs Network/Mobility
Switches** from the Actions tab (defaults reproduce this setup). To archive the
artifacts of any finished run into this directory, dispatch **Archive Run
Artifacts** with that run's ID (the number in the run's URL).
