# Single-mechanism ablation A — legacy own-prior confirmation reference

Isolates the **exploiter network-confirmation change** of the 2026-08-25
mechanism revision: identical to `results-mechfix/plots-config-switches/`
except `confirmation_reference='own'` (legacy own-prior confirmation
reward). Stochastic report rounding is KEPT.

- **Run**: [32839355312](https://github.com/tinacomes/DisasterAI/actions/runs/32839355312),
  2026-08-25, workflow *Run Primary Alignment Sweep only*, code `2b2ffd7`
  (= `30f89e0` mechanisms), main-model configuration (mobility=1,
  `network_type=spatial_bridged`, `query_scope=network`), 11 α × 20
  seed-paired replications × 200 ticks.

**Result: nearly indistinguishable from the full revision.** α* = 0.6 for
8/12 composites (full revision: 7/12), all key series within seed noise of
`results-mechfix` (e.g. α=1: SECI −0.26/−0.30 vs −0.24/−0.33 per type;
combined MAE 1.79 vs 1.82; unmet 2.32 vs 2.86). Conclusion: the network
confirmation reference has a negligible aggregate effect — the trusted
network's consensus mostly coincides with the member's own prior — so its
value is the theoretical coherence of mechanism (ii), and the restored
interior α* is attributable to the dose-linearisation (see
`results-ablation-detround/`).
