# Ablation — Legacy Consensus Targeting on the Fixed Model

Phase-1 ablation isolating the **confirmation-target mechanism**: identical to
the verification sweep in `results-verification/` except that the AI's
confirmation target is switched back to the legacy rule
(`confirmation_target=consensus`: trusted-network consensus for exploitative
callers, own prior for exploratory callers). Everything else — the explorer
rejected-remote verification fix and the per-source Q/trust batching — is the
fixed code, so any difference from `results-verification/` is attributable to
the targeting rule alone.

- **Run**: [32338797843](https://github.com/tinacomes/DisasterAI/actions/runs/32338797843),
  2026-08-20, workflow **Compare Baseline vs Network/Mobility Switches**.
- **Code version**: commit `55d4b2b` (fixed model) with
  `confirmation_target=consensus`.
- **Parameters**: 11 α levels × 20 replications × 200 ticks, replicate *i*
  seeded with seed *i* — seed-paired with `results-verification/` (and, by seed
  index, with the legacy run in `results/`).

## Directories

Same layout as `results-verification/`: `plots-config-baseline/`,
`plots-config-switches/` (figures, summary tables, `experiment_results.json`
including the `lockin_*` and `l1pool_*` series), `config-comparison/`
(cross-config paired-delta tables).

## Ablation verdict

**The legacy consensus targeting is behaviourally near-inert on the fixed
model.** Seed-paired per-α deltas (consensus − individual, last 75 ticks) are
statistically indistinguishable from zero for SECI (both types), MAE, AI query
share, AI trust, lock-in, and L1+ pool size in both configurations; the few
isolated CI exclusions (e.g. ΔSECI_exploit +0.09 at α=0.7 switched — the
*opposite* sign to the echo-chamber-amplification hypothesis) are scattered,
small, and consistent with multiple testing across 11 α × many metrics. The
interior α\* is preserved (switched spread: consensus [0.2–0.4] vs individual
[0.3–0.6], all interior).

Mechanistic explanation (see instrumented probes in the session record): the
consensus differs from the caller's own prior on only ~10–18% of
exploiter-reported cells (mean gap ≈ 1.5 levels), is no closer to ground
truth, and the exploiters' own reward channel scores confirmation against
their *own stored prior* regardless of what the AI targets — so retargeting
the report barely changes acceptance, rewards, or beliefs.

Two consequences:

1. All qualitative findings of the paper are invariant to the confirmation
   target; the type-agnostic `individual` rule (the default) is the right
   specification and costs nothing empirically.
2. Where the legacy archived run (`results/`) differs from either fixed-model
   run (e.g. exploitative AI query share 0.34→0.60 across α in the legacy run
   vs 0.45→0.67 here), the difference traces to the **reward-channel fixes**
   (explorer rejected-remote scoring, Q/trust batching), not to the targeting
   rule.
