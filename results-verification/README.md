# Verification Sweep — Type-Agnostic AI (fixed model)

Results of the **verification sweep** for the model fixes introduced in commit
`55d4b2b` (type-agnostic confirmation target, explorer rejected-remote
verification, per-source Q/trust batching, AECI-LockIn and L1+ pool metrics).

- **Run**: [32298278561](https://github.com/tinacomes/DisasterAI/actions/runs/32298278561),
  2026-08-19, workflow **Compare Baseline vs Network/Mobility Switches**.
- **Code version**: commit `55d4b2b` (branch `claude/pnas-paper-ai-behavior-ebovfd`).
- **Model configuration**: `confirmation_target=individual` (the new default —
  the AI targets the caller's own prior for BOTH agent types and never
  conditions on the caller's cognitive type).
- **Parameters**: 11 α levels (0.0–1.0), 20 replications per α, 200 ticks,
  seed-paired across configurations — identical to the canonical archived run
  32040117179 in `results/`.

## Directories

| Directory | Contents | Configuration |
|---|---|---|
| `plots-config-switches/` | Figures, `summary_table.csv/md`, `experiment_results.json` | mobility = 1, `network_type = spatial_bridged`, `query_scope = network` |
| `plots-config-baseline/` | Same file set | mobility = 0, `network_type = components`, `query_scope = global` |
| `config-comparison/` | Cross-config paired-delta tables and overlay figure | baseline vs switches |

`experiment_results.json` additionally contains the new per-type series
`lockin_*` (AECI-LockIn) and `l1pool_*` (mean L1+ beliefs per agent).

## Headline comparison with the canonical legacy run (`results/`)

All qualitative findings **survive** the type-agnostic fix (steady state =
last 75 ticks, switched configuration unless noted):

- **Explorer-driven SECI deepening**: seci_explor −0.010 (α=0) → −0.326 (α=0.9),
  vs −0.011 → −0.295 in the legacy run. The deepening never depended on the
  legacy consensus targeting.
- **Structural precondition (Finding 1)**: paired per-seed ΔSECI_exploit
  (switches − baseline) at α=0.9/1.0 = −0.29/−0.34 (95% CI excludes 0).
  Under global access the exploitative echo chamber dissolves at high α
  (baseline seci_exploit ≈ +0.02); under network-gated access it persists
  (≈ −0.27/−0.31).
- **Exploiter AI capture**: exploitative AI query share rises 0.47 → 0.70
  across α (legacy 0.34 → 0.60) — the sycophancy-capture mechanism is
  stronger under individual targeting, as the seed-paired counterfactuals
  predicted.
- **Accuracy collapse**: MAE per type matches the legacy run almost exactly
  (explorers 0.55 → 1.75 across α; exploiters flat ≈ 1.8).
- **Interior α\***: composite-dependent spread narrows to [0.3, 0.6]
  (legacy switched spread was wider); all six composites give an interior
  optimum.
- **New diagnostics**: explorer L1+ belief pool collapses 114 → 27 beliefs
  per agent from α=0 to α=1 (belief starvation under confirming AI), and
  explorer AECI-LockIn deepens −0.04 → −0.16 (AI-heavy explorers' beliefs
  freeze relative to AI-light peers). Exploiter LockIn is positive at low α
  (correction churn via relief feedback) — read as an α-gradient per type,
  see `calculate_aeci_lockin` in `DisasterAI_Model.py`.

These results are the reference for the fixed model; the legacy run in
`results/` remains the record of the pre-fix code (reproducible with
`confirmation_target=consensus` plus code version `0e05139`).
