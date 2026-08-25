# Single-mechanism ablation B — legacy deterministic report rounding

Isolates the **dose-linearisation change** of the 2026-08-25 mechanism
revision: identical to `results-mechfix/plots-config-switches/` except
`report_rounding='deterministic'` (legacy `np.round`, near-step effective
α). The network confirmation reference is KEPT.

- **Run**: [32839367900](https://github.com/tinacomes/DisasterAI/actions/runs/32839367900),
  2026-08-25, workflow *Run Primary Alignment Sweep only*, code `2b2ffd7`
  (= `30f89e0` mechanisms), main-model configuration (mobility=1,
  `network_type=spatial_bridged`, `query_scope=network`), 11 α × 20
  seed-paired replications × 200 ticks.

**Result: the composite-robust interior α\* degrades without the rounding
fix.** α* fragments to spread [0.2, 0.6] with only 4/12 composites at 0.6
(full revision: 7/12 at 0.6; ablation A: 8/12). Mid-range dosing distorts
the curves (e.g. exploiter SECI −0.42 at α=0.5 vs −0.36 under stochastic
rounding; combined MAE 1.33 vs 1.28), because deterministic rounding
delivers almost no confirmation below α≈0.5 and saturates above α≈0.9.
Conclusion: the stochastic-rounding dose linearisation is the primary
driver of the restored, composite-invariant interior optimum; the
population-level composites are the most robust to the ablation (still
α* = 0.6 here).
