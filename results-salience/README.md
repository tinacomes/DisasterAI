# M7 salience experiment — C12 decision + exploiter confirmation trap

Salience-weighted evaluation sweep on the revised model:
`salience_weight` ∈ {0, 0.5, 1} × α ∈ {0.0, 0.3, 0.7, 0.8, 0.9, 1.0},
main-model configuration, N=20 seed-paired replications, 200 ticks.
Since commit `30f89e0`, `salience_weight` scales BOTH channels: the
explorers' verified-report evaluations and the exploiters'
confirmation-channel learning rate — so one sweep answers both questions
below. α = 0.0/0.3 were added to the M7 grid for the exploiter readout.

- **Run**: [32852587340](https://github.com/tinacomes/DisasterAI/actions/runs/32852587340),
  2026-08-25, workflow *Run Salience Experiment (M7)*, code `bbfb5ec`.
- `salience-tables/` holds the per-cell late-run tables
  (`robustness_tables.md`, `salience_summary.txt`); `salience-<s>-<α>/`
  hold the per-cell result JSONs.

## Readout 1 — C12 (explorer trust in confirming AI): **stands at every salience level**

Explorer AI trust declines only marginally faster under full salience:
0.882 → 0.835 across α at s=0 (the C12 base-rate finding) vs
0.899 → 0.823 at s=1. Even when an error about a disaster cell carries 6×
the weight of a confirmed empty cell, explorers never strongly punish a
confirming AI through trust — their AI query share is actually *higher*
under salience (0.57–0.67 at s=1 vs 0.47–0.61 at s=0), because severity
weighting also rewards the AI's superior coverage of disaster cells.
**Decision (M7): keep `salience_weight = 0` mainline; C12 is a robust
finding, not an artifact of uniform evaluation.**

## Readout 2 — exploiter confirmation trap: salience removes the capture gradient, but by SOCIAL RETRENCHMENT, not by truth-seeking

At s=0 the exploiters' AI query share climbs 0.534 → 0.698 across α (the
capture mechanism; the trap at α=0 is base-rate diluted — a truthful AI
agrees with an exploiter on most reported cells). At **s=1 the gradient
disappears**: AI share is flat at 0.48–0.51 for every α, and exploiter AI
trust drops to ≈0.42 throughout. But the exploiters do not become more
accurate — they substitute toward their network: at the truthful endpoint
(α=0) their social echo chamber **deepens** (SECI_exploit −0.521 at s=1
vs −0.371 at s=0) and their relief precision **falls** (0.427 vs 0.574),
with MAE unchanged (≈1.8 at all cells). Making disconfirmation salient
ejects confirmation-seekers from the one truthful channel and pushes them
back into the social bubble — the capture mechanism of Finding 3 is
contingent on base-rate-diluted evaluation, and its removal is not a
remedy.

## Secondary observations

- Explorer AI-side echo indices deepen faster with α under salience
  (AECI-IE-chan_er at α=1: −0.28 (s=0) → −0.50 (s=1)) while explorer MAE
  worsens slightly mid-range (1.25 vs 1.08 at α=0.7) — heavier weighting
  of disaster cells narrows what the explorers effectively consume.
- The operational U-shape is retained at all salience levels (unmet needs
  minimum at α=0.7: 0.38 / 0.19 / 0.35 for s=0/0.5/1); s=1 slightly
  softens the α=1 collapse (2.61 vs 3.15 unmet).
- s=0.5 is intermediate and adds no qualitative change.
