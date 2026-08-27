# M9 alpha-reversal (hysteresis) experiment — LOCAL PROTOTYPE (N=6)

> **SUPERSEDED (2026-08-27):** the canonical N=20 × 300-tick run is
> archived in [`results-reversal/`](../results-reversal/) (run
> 33047514582). Cite from there; this directory remains only as the
> prototype provenance.

Reduced local prototype of the M9 experiment
([`experiments/alpha_reversal.py`](../experiments/alpha_reversal.py)):
**α = 1.0 → 0.0 switched at tick 100** versus the constant α=1.0 and
α=0.0 anchors on the same seeds (0–5), 250 ticks, main-model
configuration. The publication-grade run is the 6-condition × N=20 ×
300-tick matrix of
[`run-alpha-reversal.yml`](../.github/workflows/run-alpha-reversal.yml)
(adds 0.8→0.0, the 0.0→1.0 late-onset probe, and the 0.8 anchor).

## Prototype verdict: the chamber unwinds; the starvation does not (yet)

Repairing the AI policy mid-run is **partially** effective — the
community-level chamber is reversible, the knowledge damage has a long
tail:

- **Chamber: reversible.** The accuracy-seekers' chamber, standing at the
  switch in 5/6 seeds, dissolves after the repair in 4/5 (median lag
  ≈ 20 ticks to sustained recovery); under the constant α=1.0 anchor it
  dissolves in 0/5. Endpoint SECI_explor returns to the truthful-anchor
  level (−0.01 vs −0.01; α=1 anchor: −0.30).
- **Starvation: long tail.** 150 ticks after the repair the L1+ belief
  pool has recovered only part-way (93 vs 118 truthful anchor / 29
  aligned anchor) and belief error is still elevated (MAE_explor 0.62 vs
  0.45 / 1.66) — both series are still converging at the horizon.
  Roughly: the beliefs starved during 100 aligned ticks take longer than
  150 truthful ticks to regrow.
- **Behavior: reversible.** Capture reverses (the confirmation-seekers'
  AI query share falls back to the truthful-anchor trajectory), explorer
  precision recovers (0.99), and the spatial aid gap returns near the
  truthful anchor (−4.1 vs −3.0; α=1 anchor −10.1).

Paper framing this supports (pending the N=20 CI run): the *social*
echo chamber is a property of the ongoing policy and dissolves when the
policy is repaired, but the *epistemic* damage — the emptied belief
pool and its accuracy cost — outlives the sycophancy that caused it by
at least the duration of the exposure. Without repair (constant α=1),
none of it unwinds within the horizon.

## Files

- `reversal_1.0_to_0.0.json`, `reversal_1.0_to_1.0.json`,
  `reversal_0.0_to_0.0.json` — per-seed trajectories (curated series),
  N=6 each.
- `tables/reversal_summary.md` — per-seed recovery + endpoint table.
- `tables/reversal_trajectories.png` — seed-paired mean ± 95% CI
  trajectories with the switch marked.

## Caveats

N=6 seeds and a 250-tick horizon: CIs are wide (especially the spatial
gaps) and the starvation series had not converged at the horizon —
treat every number as provisional until the workflow run replaces this
directory's role. Seeds are the canonical 0–5 subset, so the future
N=20 run extends (not replaces) these replicates.
