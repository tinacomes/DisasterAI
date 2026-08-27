# Lifecycle (dynamics) metrics — PROTOTYPE post-processing layer

Derived from the canonical archived trajectories in
`../plots-config-switches/experiment_results.json` (main model) and
`../plots-config-baseline/experiment_results.json` (control) by
[`tools/lifecycle_metrics.py`](../../tools/lifecycle_metrics.py). No new
simulation: everything here is a functional of the already-archived
cross-replication **mean** trajectories of run 32821105202.

## Why this layer exists

The paper's Results are steady-state endpoints; its language ("silently
starves", "progressively captures") is dynamic. This layer quantifies the
dynamics with the thresholds the archived lifecycle/transition figures
already use (formation SECI < −0.1; dissolution sustained > −0.05):

- **Chamber episodes** per type/α/config: formation tick, peak depth, every
  formation–dissolution episode (chambers are two-wave for the
  confirmation-seekers: early chamber → mid-run recovery → re-deepening),
  final dissolution tick (right-censored at tick 200), persistence
  fraction, standing-at-end flag.
- **Capture onset**: first sustained AI-majority tick per type.
- **Mechanism cascade**: half-transition tick per mechanism variable
  (capture → starvation → chamber → precision → periphery gap).
- **Final-window slopes**: stationarity check — non-zero slope at the
  horizon means the outcome was still accumulating when the run ended.

## Headline prototype readings (main model)

1. **Alignment does not deepen chambers — it prevents them from
   dissolving.** Accuracy-seeking chambers form at tick ≈ 15–25 at every
   α ≤ 0.7 with peak depth varying little (−0.35 … −0.51); at α ≤ 0.5 they
   dissolve by tick 100–120, from α ≥ 0.6 they never dissolve within the
   200-tick horizon. The irreversibility threshold coincides with α*.
2. **The confirmation-seekers' chamber is two-wave**, and only unrestricted
   access at α ≥ 0.9 dissolves it for good (control: final dissolution at
   tick 95–175); under network-bounded access it is standing at the end at
   every α.
3. **Capture accelerates ~50×**: sustained AI-majority onset for
   confirmation-seekers moves from tick ≈ 55 (α = 0) to tick ≈ 1–5
   (α ≥ 0.7).
4. **Harms had not equilibrated at the horizon at α = 1**: spatial MAE-gap
   slope +0.15 / aid-gap slope −1.97 per 100 ticks over the final 50 ticks
   (≈ 0 at low α) — the reported steady-state periphery harms are lower
   bounds.

## Files

- `lifecycle_metrics.md` / `.csv` — full tables (both configurations).
- `proto_lifecycle_timeline.png` — chamber lifespan bars per α/type/config.
- `proto_depth_vs_persistence.png` — peak depth vs final dissolution vs
  persistence (the depth-flat / dissolution-blocked contrast).
- `proto_trajectories.png` — SECI trajectories at α ∈ {0, 0.6, 1}.
- `proto_mechanism_cascade.png` — half-transition ordering at α ∈ {0.8, 1}.

## PROTOTYPE status — what these numbers are NOT

Computed from mean trajectories: the archived JSONs carry per-seed values
only at steady state, so none of these numbers has a confidence interval
and none is seed-robustness tested. Before any of this enters the
manuscript, re-dispatch the primary sweep with per-seed lifecycle columns
(metrics are observation-only: on the same seeds every existing series
reproduces bit-identically and the run only gains columns) and treat the
per-seed lifecycle values with the same machinery as the endpoint claims
(paired deltas, Holm; survival analysis for the censored dissolution
times). AECI-LockIn is excluded from the cascade because its
initialisation transient defeats half-crossing timing at mean-trajectory
resolution.
