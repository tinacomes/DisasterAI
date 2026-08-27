# M9 alpha-reversal (hysteresis) experiment — CANONICAL (N=20): PASS, with a two-part verdict

Full run of the M9 experiment
([`experiments/alpha_reversal.py`](../experiments/alpha_reversal.py)):
**α switched at tick 100** (1.0→0.0, 0.8→0.0 repair; 0.0→1.0 late-onset
probe) versus constant-α anchors (0.0, 0.8, 1.0) on the **same seeds
0–19**, 300 ticks, main-model configuration. Supersedes the N=6 local
prototype in [`results-reversal-prototype/`](../results-reversal-prototype/).

- **Run**: [33047514582](https://github.com/tinacomes/DisasterAI/actions/runs/33047514582),
  2026-08-27, workflow *Run Alpha Reversal Experiment (M9)*, dispatched
  from `main` at `46204f7`.
- Per-seed trajectory JSONs in `reversal-*/`; summary table and
  seed-paired trajectory figure in `reversal-tables/`.

## Verdict 1 — the chamber is a property of the ongoing policy: repairing the AI dissolves it

With the policy switched to truthful at tick 100, the accuracy-seekers'
chamber (standing at the switch in 19/20 seeds at α=1.0, 20/20 at α=0.8)
dissolves in **18/19** (median lag 50 ticks) and **20/20** (median 30).
Under the constant α=1.0 anchor it dissolves spontaneously in only
**4/19 within 300 ticks** — the 200-tick "never dissolves" reading of the
lifecycle layer is horizon-robust at α=1.0. At α=0.8 the longer horizon
softens it: 13/20 anchor seeds eventually dissolve, so at intermediate-
high alignment the chamber is very-slowly-self-limiting rather than
permanent, while at full alignment it is genuinely locked. Endpoint
SECI returns to the truthful-anchor level in both repair conditions
(−0.011 ± 0.009 and −0.010 ± 0.008 vs −0.010 ± 0.011; constant α=1.0:
−0.232 ± 0.062). Capture also reverses: the confirmation-seekers' AI
query share falls back onto the truthful-anchor trajectory.

## Verdict 2 — the starvation has a long tail: the epistemic damage outlives the policy

200 truthful ticks after the α=1.0→0.0 repair, the accuracy-seekers'
belief pool has regrown only to **102.0 ± 5.9 of the truthful anchor's
120.2 ± 6.3**, and their belief error is still elevated (MAE
**0.486 ± 0.033 vs 0.412 ± 0.029**; α=1.0 anchor 1.592 ± 0.103) — CI-
separated from the truthful anchor at the horizon, and the recovery
took roughly twice as long as the 100 aligned ticks that caused the
damage. Same pattern after 0.8→0.0 (pool 108.9 vs 120.2; MAE 0.465 vs
0.412). Operational quantities recover faster: explorer precision is
back to ≈1.0 and the spatial gaps return to the truthful-anchor level.

## Bonus finding — the asymmetry: a healthy history protects

The late-onset probe (0.0→1.0 at tick 100) shows the reverse switch is
NOT symmetric: 200 confirming ticks applied to an informed population
leave MAE at only 0.567 ± 0.087 (constant α=1.0: 1.592), the belief pool
at 107.4 ± 8.2 (vs 29.8), and **no re-formed accuracy-seekers' chamber**
(endpoint SECI −0.023 ± 0.014). The explorer chamber and the starvation
spiral are formation-phase phenomena: sycophancy does its damage when it
catches a population whose beliefs are still empty — exactly the
disaster-onset condition. (The confirmation-seekers' second-wave chamber
does deepen under late-onset sycophancy, and capture onsets late but
fully.)

## Paper framing this supports

The *social* echo chamber and the capture loop are properties of the
ongoing policy — repair dissolves them; the *epistemic* damage (the
emptied belief pool and its accuracy cost) outlives the policy by more
than the exposure duration; and the harm is front-loaded: alignment is
most dangerous exactly when beliefs are forming. Fits the Discussion as
the dynamic sharpening of "monitoring must target community-level
convergence": recovery of the chamber index after an intervention does
NOT certify recovery of the knowledge base.

## Files

- `reversal-{0..5}/reversal_<pre>_to_<post>.json` — per-seed curated
  trajectory series (N=20 each; seeds 0–19).
- `reversal-tables/reversal_summary.md` — recovery + endpoint table.
- `reversal-tables/reversal_trajectories.png` — seed-paired mean ± 95%
  CI trajectories, switch tick marked.

Regeneration: `python3 experiments/alpha_reversal.py collect
--results-dir results-reversal --save-dir results-reversal/reversal-tables`.
