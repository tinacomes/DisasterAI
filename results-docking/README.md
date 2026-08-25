# M6 dyadic docking — qualitative reproduction of Glickman & Sharot: PASS

One human (each type) × one AI, no network effects and no relief feedback,
seeded false epicenter (start false-belief level 2.07), 20 seeds × 40
interaction rounds × 11 α, vs a human–human control pair. Revised model
(`60c96e2` = `30f89e0` mechanisms).

- **Run**: [32866369049](https://github.com/tinacomes/DisasterAI/actions/runs/32866369049),
  2026-08-25, workflow *Run Dyadic Docking (M6)*.
- Files: `dyadic-docking/dyadic_results/dyadic_docking.json` (per-cell
  series), `dyadic_docking.png` (SI Fig. S6 basis),
  `dyadic_results_log.txt` (checks).

## Verdict — both formal checks PASS for both agent types

Final false-belief level after 40 rounds (start 2.07; lower = corrected):

| α (AI partner) | 0.0 | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 | human–human |
|---|---|---|---|---|---|---|---|
| exploratory  | **1.49** | 1.61 | 1.78 | 2.03 | 2.07 | **2.07** | 1.54 |
| exploitative | **1.97** | 2.04 | 2.07 | 2.07 | 2.07 | **2.07** | 1.50 |

1. **Bias retention grows with α** (both types): the false belief is
   corrected under a truthful AI and fully preserved under a confirming
   one, with a smooth gradient in between — the dyadic human–AI feedback
   amplification of Glickman & Sharot (2024), qualitatively reproduced.
2. **An aligned AI retains more bias than a human partner** (both types):
   at α=1 the dyad keeps the full false belief (2.07) while the
   human–human pair corrects to ≈1.5.

Type nuance worth one SI sentence: for **exploiters even a truthful AI
barely corrects** (1.97 vs the human pair's 1.50) — the D/δ acceptance
window rejects the AI's disconfirming reports, while a trusted human
partner benefits from the friend-widened acceptance threshold. The
dyadic setting thus already contains the population-level trap in
miniature.

## Caveats

- Docking is qualitative pattern-reproduction, not parameter-matched
  calibration to the Glickman & Sharot experiments; say so explicitly in
  the SI.
- The dose gradient saturates early (α≈0.4 exploiters, α≈0.7 explorers):
  with a single false belief on a 0–5 integer scale, mid-α confirmation
  suffices to fully preserve it. The population-level sweeps, not the
  dyad, carry the fine-grained dose–response.

**Consequence for the venue decision** (`PNAS_WRITING_INSTRUCTIONS.md`
§1): the M6 gate is satisfied; escalation from PNAS Nexus to PNAS Direct
Submission now hinges on M2–M5 only.
