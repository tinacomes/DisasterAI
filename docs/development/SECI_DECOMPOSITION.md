# What SECI actually measures at high α — a denominator problem

Prompted by the author's observation that "under the truthful AI, exploiters
do not just fall prey to their social bubbles." Investigating that produced a
finding that bears directly on **Finding 1**, the paper's headline.

Method: `scratchpad/variance_probe.py` re-derives SECI from its raw parts
instead of reading the finished index — global L1+ belief variance (the
denominator), community L1+ belief variance (the numerator), and a third
quantity the index does not contain: the **spread of member beliefs**, the SD
across a community's members of each member's own mean L1+ belief. That third
number answers the question "echo chamber" is actually asking: *do members of
this community agree with one another?* 3 seeds, 200 ticks, both
configurations, α ∈ {0, 0.5, 1.0}; steady-window means (last 37.5%).

## 1. The numbers

| cell | Var_global | Var_comm (exploit) | member spread (exploit) | Var_comm (explor) | member spread (explor) |
|---|---|---|---|---|---|
| control, α=0   | 0.952 | 0.493 | 0.260 | 0.978 | 0.133 |
| control, α=0.5 | 0.851 | 0.509 | 0.305 | 0.880 | 0.178 |
| **control, α=1.0** | **0.252** | **0.643** | **0.532** | **0.182** | 0.121 |
| main, α=0      | 1.079 | 0.485 | 0.256 | 1.105 | 0.109 |
| main, α=0.5    | 1.016 | 0.557 | 0.275 | 1.024 | 0.158 |
| **main, α=1.0** | **0.610** | **0.450** | **0.296** | **0.395** | 0.149 |

## 2. The exploiter "dissolution" is a denominator artifact

Between α=0 and α=1 in the **control**:

- the SECI **denominator collapses by 74%** (0.952 → 0.252);
- the **numerator rises** (0.493 → 0.643) — exploiter communities become
  internally *more* dispersed, not less;
- **member disagreement doubles** (0.260 → 0.532).

SECI therefore turns positive not because the chamber broke, but because the
population it is compared against became degenerate. And the communities are
not converged either: their members agree with each other *less* than at any
other α. The state at α=1 under unrestricted access is neither an echo chamber
nor healthy diversity — it is **atomization**: every agent served their own
prior, each in a private bubble, with no shared community position at all.

Why the global variance collapses: at α=1 the accuracy-seekers are starved
(L1+ pool 109 → 19 in the control) into near-identical, mostly-empty beliefs.
They make up half the population, so the pooled global variance they dominate
crashes — and SECI divides by it.

The main model shows the same mechanism, muted: denominator −43%, numerator
flat, spread +16%. Mobility keeps first-hand observation flowing, so neither
the reference population nor the communities degenerate as far.

## 3. Where SECI *does* work: the explorers

The explorer columns are the control condition for this critique. There the
**numerator genuinely collapses** (control 0.978 → 0.182, −81%; main 1.105 →
0.395, −64%) while member spread stays flat. Members really do converge on a
shared position, and it is a wrong one (MAE 1.94). That is a real echo
chamber, correctly registered.

So SECI is not broken as a construct. It fails specifically when the
treatment moves its denominator — the same class of defect as AECI-Var's
truth-convergence confound, which M1 was created to remove. We removed it on
the AI side and left it in place on the social side.

## 4. Consequence for the manuscript

**Finding 1's control arm as currently written is not supported.** "Under
unrestricted access, high alignment dissolves the confirmation-seekers' echo
chamber" describes a ratio, not a state. The defensible statement is that
unrestricted access lets a confirming AI **atomize** those communities —
members disagree more, and the population-level reference collapses — whereas
network-gated access preserves a shared (wrong) community position. That is
arguably a *more* interesting result, and it keeps the structural contrast
that carries the paper, but it is a different claim and it is the author's to
make.

The paired ΔSECI numbers themselves are unaffected as *measurements*; what
changes is what they may be said to mean.

## 5. Recommended reporting change

Report SECI with its parts, not alone:

1. **SECI** as now, for continuity with the literature;
2. **Var_global** alongside it, so a moving denominator is visible;
3. **member spread** as the direct "do members agree?" measure — it needs no
   denominator and cannot be confounded this way.

`tools/evolution_figures.py` already plots the L1+ pool beside each index for
the same reason. Adding member spread to the model's per-tick metrics is a
small, observation-only change.

## 6. Related: the indices have not converged at T=200

Separately visible in the evolution figures: exploiter SECI oscillates
(chamber forms by tick ~20, recovers by ~75–100, re-forms by ~140, still
recovering at T=200) and is nearly α-independent in the main model, while
AECI-IE rises monotonically from ≈ −0.9 toward 0 across the entire run for
both types. The "steady state = last 75 ticks" assumption does not hold for
these series. A T=600 convergence run is in progress; if the indices settle
and the configuration contrast survives, the fix is longer runs, and if not,
the endpoint means should be replaced by trajectory features (peak depth,
recovery tick, fraction of run in-chamber) — which the NHB draft's lifecycle
analysis already provides.
