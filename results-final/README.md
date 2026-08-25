# Final Sweep — pre-revision dataset (superseded 2026-08-25)

> **Status update (2026-08-25):** this archive records the model **before**
> the mechanism revision of commit `30f89e0` (exploiter network
> confirmation reference; stochastic report rounding, which removes the
> effective-α step function documented in caveat 2 below). The
> current-model record is **`results-mechfix/`** (run 32821105202), which
> restores a composite-robust interior α* = 0.6 in the main model. Keep
> this directory for the pre-revision comparison and the ablation chain.

Results of the **primary alignment sweep on the fixed, type-agnostic model**,
extended with the M1 information-environment metrics and the co-evolution
series.

- **Run**: [32412891328](https://github.com/tinacomes/DisasterAI/actions/runs/32412891328),
  2026-08-20, workflow **Compare Baseline vs Network/Mobility Switches**.
- **Code version**: commit `36891da` (branch
  `claude/pnas-paper-implementation-aqryq7`).
- **Model configuration**: `confirmation_target=individual` — the AI targets
  the caller's own prior for BOTH agent types and never conditions on the
  caller's cognitive type.
- **Parameters**: 11 α levels (0.0–1.0), 20 replications per α, 200 ticks,
  seed-paired across configurations — identical design to `results/`
  (run 32040117179, legacy code) and `results-verification/`
  (run 32298278561, fixed code without the M1 metrics).

## Directories

| Directory | Contents | Configuration |
|---|---|---|
| `plots-config-switches/` | Figures, `summary_table.csv/md`, `experiment_results.json` | **main model**: mobility = 1, `network_type = spatial_bridged`, `query_scope = network` |
| `plots-config-baseline/` | Same file set | **control**: mobility = 0, `network_type = components`, `query_scope = global` |
| `config-comparison/` | Cross-config paired-delta tables and overlay figure | baseline vs switches |

## What is new relative to `results-verification/`

The M1 metrics are observation-only in code (they read and clear a log that
feeds nothing else), but **this run does NOT reproduce the verification run
bit-exactly on the same seeds**: the accompanying infrastructure changes in
commit `19619ee` altered the RNG consumption order, so every per-seed series
drifted (e.g. per-seed |ΔSECI| up to 0.45), while all aggregates agree
within replication noise (combined SECI at α=1: −0.312 vs −0.300; explorer
MAE 0.54 → 1.74 in both). Treat this run as a statistically equivalent
re-run with added columns, not a bit-exact reproduction. New per-type
series in `experiment_results.json`:

| Series | Construct |
|---|---|
| `aeci_ie_{exploit,explor}` | SECI's variance-ratio construct applied to the report levels the **AI channel serves** a community (belief baseline) |
| `seci_ie_{exploit,explor}` | Same over **human-delivered** reports — consistency check against SECI |
| `aeci_ie_chan_*`, `seci_ie_chan_*` | Channel baseline: community served pool vs the **global served pool of the same channel** |
| `aeci_ie_rel_*`, `seci_ie_rel_*` | Community-relative baseline: served pool vs the community's **own** beliefs (0 = the channel mirrors the community back) |
| `ai_reliance_{exploit,explor}` | AI share of all delivered external reports |
| `effective_alpha_*` | **Delivered** confirmation fraction — the manipulation check for the integer-rounding saturation of nominal α |

Each carries `_mean`, `_std` and `_ss_runs` (per-seed steady-state values,
for paired contrasts). Steady state = mean over the last 75 of 200 ticks.

## Provenance note

The artifacts were first archived into `results/` by an accidental dispatch
of *Archive Run Artifacts* with the default `dest` (commit `c891f5d`). That
commit was purely additive and overwrote nothing; the three directories above
were moved here and the 22 redundant per-α `netmob-*` artifact folders were
dropped, since their contents are already merged into each configuration's
`experiment_results.json`. `results/` retains only its legacy content.

## Caveats carried into the manuscript

1. For **exploitative** communities the belief-baseline `aeci_ie` collapses
   onto SECI at α ≥ 0.8 (per-seed r = 0.89–0.95) and is not an independent
   AI-side measurement there. Report AECI-IE per type; the exploiter
   social→AI handover claim rests on `aeci_ie_rel` → 0 together with flat-high
   `ai_reliance` and flat MAE. See `docs/development/M1_VALIDATION.md`.
2. `effective_alpha` saturates at 1.0 for α ≥ 0.9, so α = 0.9 and α = 1.0 are
   identical at the level of what is served. The operational cliff coincides
   with this delivery boundary rather than with a threshold in the policy.
   See `docs/development/COEVOLUTION_ANALYSIS.md`.
