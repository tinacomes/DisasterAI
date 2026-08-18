# S8 factor sweeps (run 32113892791, 2026-08-18)

One-at-a-time sweeps over rumour probability {0, 0.5, 1}, disaster dynamics
{0, 2, 3} and exploitative share {0.2, 0.5, 0.8} at α = 0.5; 20 replications,
100 ticks (formation phase), **main-model configuration**, pinned environment.
`factor-sweeps/` holds the per-condition JSONs, the merged
`experiment_results_with_factors.json`, and `factor_comparison.png`
(regenerated locally after fixing the figure title to read actual run counts).
Note: base parameters hold rumour probability at 1.0, so the disaster and mix
conditions are evaluated in the harshest rumour environment. See Supplementary
S8 of the paper.
