# Mechanism-revision sweep — candidate canonical dataset (2026-08-25)

Primary alignment sweep (baseline vs switches) on the model **after the
2026-08-25 mechanism revision** (commit `30f89e0`). Supersedes
`results-final/` as the current-model record; see §Attribution for what
changed and how to isolate each change.

- **Run**: [32821105202](https://github.com/tinacomes/DisasterAI/actions/runs/32821105202),
  2026-08-25, workflow **Compare Baseline vs Network/Mobility Switches**.
- **Code version**: commit `30f89e0` (branch
  `claude/explorer-exploiter-mechanisms-ymi8pl`).
- **Parameters**: 11 α levels (0.0–1.0), 20 seed-paired replications per α,
  200 ticks — identical design to `results-final/` (run 32412891328,
  pre-revision code), so the two archives are directly comparable per seed
  *within* each archive (the revision changes RNG consumption, so runs are
  NOT bit-comparable across archives).

## What changed in the model (all defaults; legacy behaviour behind flags)

1. **`confirmation_reference='network'`** — exploiters' confirmation reward
   now scores sources against the **trusted-network consensus** when
   defined (≥ 2 trusted friends with data), falling back to their own
   stored prior. Previously the consensus fed only the accuracy channel,
   which exploiters never use, so their confirmation target was always the
   own prior (mechanism (ii) was dead code). Legacy: `'own'`.
2. **`report_rounding='stochastic'`** — the AI's aligned report
   (1−α)·truth + α·belief is discretised by probabilistic rounding, making
   the **delivered** confirmation dose linear in α in expectation. The
   legacy `np.round` made effective α a near step function (≈0 for α≤0.4,
   saturated at 1.0 for α≥0.9). Legacy: `'deterministic'`.
3. **`salience_weight`** now also scales the exploiters'
   confirmation-channel learning rate (previously explorer verification
   only). Behaviour-neutral at the mainline default `0`.
4. **Population-level metrics** (observation-only): `seci_pop`,
   `aeci_ie_pop` / `seci_ie_pop` (belief- and channel-baseline),
   `lockin_pop`, population composites in the α* table, and the
   `population_evolution.png` figure (full trajectories: population +
   per-type, all α).

## Headline results (late-run means; negative = echo chamber)

- **The interior optimum is restored and composite-robust in the main
  model**: α* = **0.6** for 7 of 12 composite definitions in the switches
  configuration — including both population-level composites, the
  channel-baseline pair (SECI + AECI-IE-chan, ± MAE), and SECI-only — with
  the remaining variants at 0.1–0.3. In the control the bubble composites
  land at 0.8–0.9 (interior; the pre-revision corner solution α* = 1.0 is
  gone because SECI re-deepens and AECI-IE deepens at α = 1).
- **The dose–response is smooth**: combined MAE rises gradually at every α
  step (1.17 → 1.82 switches; 1.25 → 1.96 baseline) with no jump at α=0.5
  or cliff at α=0.9 — the previous cliff was the delivery step function,
  not a behavioural threshold.
- **Per type (fragmentation)**: in the switches model the exploitative
  chamber persists at all α (SECI_exploit −0.39 … −0.18) while under
  global access it dissolves (−0.45 → +0.05); the explorer series deepens
  0 → −0.33 with α in both configurations. Explorer MAE 0.54 → 1.74
  (switches): the accuracy cost of alignment still falls on the
  accuracy-seekers.
- **Population (societal)**: SECI-pop is −0.22 → 0.00 in the control —
  the population lens *masks* the exploiter-dissolve/explorer-deepen
  crossover, which is why both levels are reported. The population
  channel-baseline AI index AECI-IE-chan-pop is **U-shaped in α in both
  configurations** (shallowest ≈ −0.15/−0.19 at α=0.7; deepest at the
  endpoints), i.e. the served information environment is most diverse at
  intermediate alignment.
- **Operational U-shape intact**: unmet needs 1.59 → 0.25 (α=0.6) → 2.86
  (switches); 2.23 → 0.49 (α=0.6) → 10.18 (baseline). Paired per-seed
  deltas: the switches configuration reduces MAE and raises precision at
  every α, and cuts unmet needs by −7.3 cells at α=1 (95% CIs exclude 0).

## Attribution — settled by the single-mechanism ablations (2026-08-25)

This run changes mechanisms (1) and (2) jointly relative to
`results-final/`. The two single-mechanism ablations (main-model
configuration, same design) attribute the effect:

| Dataset | Reference | Rounding | α* = 0.6 composites | Verdict |
|---|---|---|---|---|
| `results-mechfix/` (this run, switches) | network | stochastic | 7/12 | full revision |
| `results-ablation-ownref/` | **own** | stochastic | 8/12 | ≈ identical to full revision |
| `results-ablation-detround/` | network | **deterministic** | 4/12 (spread [0.2, 0.6]) | interior α* degrades |

**The stochastic-rounding dose linearisation is the primary driver of the
restored, composite-robust interior α\*.** The network confirmation
reference has a negligible aggregate effect (the trusted network's
consensus mostly coincides with the member's own prior) — its value is the
theoretical coherence of mechanism (ii), not the curve shapes. The
population-level composites are the most ablation-robust (α* = 0.6 in all
three datasets). The salience counterfactual for the confirmation trap
(both agent types) remains `salience_weight > 0`.

## Directories

| Directory | Contents | Configuration |
|---|---|---|
| `plots-config-switches/` | Figures incl. `population_evolution.png`, `summary_table.csv/md`, `experiment_results.json` | **main model**: mobility=1, `network_type=spatial_bridged`, `query_scope=network` |
| `plots-config-baseline/` | Same file set | **control**: mobility=0, `network_type=components`, `query_scope=global` |
| `config-comparison/` | Cross-config paired-delta tables, α* sensitivity, overlay figure | baseline vs switches |
