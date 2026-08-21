# M1 validation — AECI-IE / SECI-IE (information-environment echo indices)

> **SUPERSEDED IN PART (2026-08-21).** The per-type verdict below was
> derived from local runs of 3–4 seeds at 150 ticks. It does **not**
> replicate on the citable N=20 / 200-tick dataset (`results-final/`,
> run 32412891328), and the N=20 result is authoritative. Corrected
> per-type picture, main model, AECI-IE (belief baseline):
>
> | | α=0 | deepest | α=0.9 | α=1.0 |
> |---|---|---|---|---|
> | exploit | −0.183±0.037 | **−0.457** (α=0.5) | −0.222 | −0.164 |
> | explor  | −0.020±0.017 | −0.089 (α=0.5) | −0.065 | −0.055 |
>
> So the dose–response lives in the **confirmation-seekers** (seed-robustly
> below their own α=0 value at every level 0.1–0.8, paired within-config
> CIs excluding zero), reverting at α≥0.9 where delivery saturates — the
> reverse of the assignment stated below. Accuracy-seekers are flat near
> zero in the main model; in the control they collapse to −0.377/−0.351 at
> α≥0.9. Section 4's decision question stands, but option (a) must be read
> with the types swapped. The mechanism analysis in sections 3 and the
> smoke-test results are unaffected.

**Status: implementation complete and verified observation-only; the
validation criteria pass per-type for explorers and for the SECI-IE
consistency check, but FAIL for exploiters under both normalization
variants. Author decision required on the paper's primary AI-side index
(see §4).**

Setup: local steady-state check, 150 ticks, seeds {0, 1, 2}, α ∈ {0.0, 0.3,
0.5, 0.7, 1.0}, both configurations (control = immobile/components/global;
main = mobility/spatial_bridged/network), `rumor_probability=1.0`,
late-run mean of the last 50 ticks. Definitive numbers come from the N=20
CI sweep (`results-final`); the qualitative pattern below was stable across
two independent local runs (2 and 3 seeds).

Both variants apply SECI's variance-ratio formula, asymmetric [−1,1]
normalization, L1+ filter, per-community pooling, and per-type averaging to
the report levels community members receive from a channel per 5-tick
window. They differ only in the denominator:

- **AECI-IE** (belief baseline; the PNAS-brief formula): community served
  pool vs. **global belief variance**.
- **AECI-IE-chan** (channel baseline; the strict SECI parallel): community
  served pool vs. the **global served pool of the same channel**.

## 1. Results (AI channel, late-run means)

| config | type | variant | α=0.0 | α=0.3 | α=0.5 | α=0.7 | α=1.0 |
|---|---|---|---|---|---|---|---|
| control | exploit | AECI-IE      | −0.67 | −0.76 | −0.67 | −0.47 | **+0.06** |
| control | exploit | AECI-IE-chan | −0.77 | −0.80 | −0.56 | −0.41 | **−0.03** |
| control | explor  | AECI-IE      | −0.04 | +0.01 | −0.10 | −0.22 | −0.50 |
| control | explor  | AECI-IE-chan | −0.18 | −0.06 | −0.01 | −0.16 | −0.70 |
| main    | exploit | AECI-IE      | −0.36 | −0.51 | −0.45 | −0.45 | **−0.03** |
| main    | exploit | AECI-IE-chan | −0.60 | −0.55 | −0.41 | −0.45 | −0.40 |
| main    | explor  | AECI-IE      | −0.16 | −0.20 | −0.22 | −0.19 | −0.57 |
| main    | explor  | AECI-IE-chan | −0.39 | −0.28 | −0.17 | −0.18 | −0.76 |

Smoke tests (deterministic constructions, `test_confirmation_target.py`):
AECI-IE = −1.000 for a fully converged community served by an α=1 AI;
AECI-IE-chan = −1.000 for two same-type communities converged on different
levels; |AECI-IE| < 0.5 for a truthful AI serving scattered queries.

## 2. Verdict against the M1 validation criteria

1. **"AECI-IE ≈ 0 at α=0"** — **PASSES for explorers** (belief baseline:
   −0.04 control / −0.16 main). **FAILS for exploiters** under both
   variants (−0.36 … −0.77 at α=0).
2. **"Monotonically negative in α in both configurations"** — **PASSES for
   explorers** (both variants, both configurations, modest mid-range
   non-monotonicity in the chan variant). **FAILS for exploiters**: the
   index is *shallowest* at α=1 (+0.06 control / −0.03 main, belief
   baseline).
3. **"SECI-IE tracks SECI per type"** — **PASSES.** Exploiters: SECI-IE
   −0.28→−0.06 vs SECI −0.35→+0.06 across α (control; main analogous).
   Explorers: SECI-IE tracks the α-gradient with a constant deeper offset
   (homophilous friend-querying filters served info more than beliefs).
4. **"α\* interior with the new composite"** — to be re-derived from the
   N=20 sweep (`Table S2` machinery reports all variants side by side).

## 3. Why the exploiter index fails — mechanism, not bug

Two structural effects, both verified by construction tests:

- **Query concentration (α=0 confound).** Exploiters query their believed
  epicenter with radius 2. The truth inside one small neighborhood is
  spatially correlated → the served pool is narrow at *any* α, so even a
  perfectly truthful AI reads as an "echo chamber" relative to global
  beliefs (belief baseline) or relative to the global served pool that
  explorers' scattered uncertainty-queries diversify (channel baseline).
  The narrowness of confirmation-seekers' information diet is genuinely
  self-selected exposure — but it is not an AI *alignment* dose-response.
- **Collapse toward SECI (α=1 confound).** At α=1 the AI serves each
  caller's own priors, so the served pool becomes a sample of the
  community's own belief pool and AECI-IE converges to SECI by
  construction: ≈ +0.06 in the control (where unrestricted access has
  *dissolved* the exploiter chamber) — the AI-side index inherits the
  social state instead of exposing the individualized bubble.

For explorers, neither confound binds (scattered queries; diverse priors
until starvation sets in), which is why the intended dose–response appears
exactly there — consistent with Finding 2's claim that the accuracy cost of
alignment falls on accuracy-seekers.

## 4. Decision required (author)

The brief's Finding-2 sentence — "the AI-channel echo index (AECI-IE)
deepens with α while ≈0 at the truthful endpoint" — is empirically true
**as a per-type statement about explorers**, and per-type reporting is the
brief's own default. Options:

- **(a) Per-type AECI-IE (recommended):** report explorer AECI-IE as the
  dose–response evidence; report exploiter AECI-IE with the
  query-concentration mechanism named (self-selected narrow exposure at all
  α; convergence toward SECI at α=1). No further code needed; both variants
  are in the citable dataset.
- **(b) Truth-matched baseline:** compare served levels against the truth
  at the *same served cells* (index 0 at α=0 by construction; isolates the
  AI's α-narrowing from query narrowing). Cleanest dose–response but
  departs from the SECI-identical construct, weakening the
  `total_bubble = |SECI| + |AECI-IE|` symmetry claim.
- **(c) Keep the pooled index and drop the ≈0-at-α=0 property** from the
  text, reporting the α=1 collapse-toward-SECI as the AI-side result.

The composite/α\* sensitivity table reports variants under (a) and both
normalizations, so the N=20 dataset supports any of these choices without
re-running.

## 5. N=20 × 200-tick paired sweep (run 32404133354) — type-averaged check

The full paired sweep (baseline = control, switches = main; 11 α levels,
20 seeds, 200 ticks) completed on commit 431dde0. Type-averaged AECI-IE
confirms the local per-type diagnosis: a modest dose-response into
mid-α (baseline −0.10 → −0.26 at α=0.5; switches −0.11 → −0.27) that
*shallows* again at α≥0.9 (−0.17 / −0.12) — exactly the signature of the
explorer deepening being cancelled by the exploiter collapse-toward-SECI.
SECI-IE tracks SECI at low-mid α and deepens sharply at α≥0.9
(baseline −0.45 vs SECI −0.15; switches −0.40 vs −0.31), consistent with
human-channel pool starvation plus convergence.

Two headline results are seed-robust at N=20 (95% CI excludes 0):
ΔSECI (switches − baseline) = −0.157 [−0.218, −0.095] at α=0.9 and
−0.151 [−0.218, −0.084] at α=1.0 (Finding 1), and the operational cliff
at α≥0.9 (baseline unmet needs 0.3–2.3 → 9.7–10.1; precision 0.70 → 0.22),
which coincides with the integer-rounding saturation of the alignment
formula (α≥0.9 serves the caller's prior exactly — see the effective-α
manipulation check).

α\* is NOT robustly interior for type-averaged composites at N=20
(SECI+AECI-IE: 1.0 baseline / 0.3 switches; +MAE: 0.1 / 0.0; chan
variants: 0.8 / 0.5 and 0.5 / 0.3) — the exploiter non-monotonicity
pollutes the average. An interior α\* claim should wait for the per-type
metric decision (§4) and be framed per type.

Note: this run predates the co-evolution metrics (aeci_ie_rel,
ai_reliance, effective_alpha). Re-dispatching the same workflow from
current main reproduces every existing column bit-identically and adds
the new ones; do that before archiving as `results-final` if those
columns should be citable.
