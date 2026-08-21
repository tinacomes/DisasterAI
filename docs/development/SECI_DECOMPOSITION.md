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

## 6. T=200 is not steady state — it is the trough of a transient

A T=600 run (both configurations, α ∈ {0, 1}, 2 seeds) settles this. The
system **does** reach a steady state, but it reaches it around **tick
250–300**, and the reported "last 75 ticks of 200" window sits squarely on
the deepest trough of the transient that precedes it.

Window means (2 seeds; T=200 window = ticks 125–200, T=600 window = 475–600):

| cell | metric | T=200 window | T=600 window | drift |
|---|---|---|---|---|
| main, α=0   | SECI_exploit | −0.458 | −0.119 | **+0.339** |
| main, α=1   | SECI_exploit | −0.438 | −0.021 | **+0.417** |
| control, α=0 | SECI_exploit | −0.398 | −0.097 | **+0.301** |
| control, α=1 | SECI_exploit | +0.102 | +0.105 | +0.003 |
| main, α=1   | SECI_exploratory | −0.342 | −0.136 | **+0.206** |
| control, α=1 | SECI_exploratory | −0.406 | −0.158 | **+0.248** |
| main, α=1   | AECI-IE_exploit | −0.159 | +0.040 | +0.199 |
| control, α=0 | L1+ pool exploit | 9.8 | 19.8 | +10.0 |

Three consequences.

**(a) The Finding-1 contrast shrinks four-fold.** ΔSECI_exploit (main −
control) at α=1 is −0.540 on the T=200 window and **−0.126** on the T=600
window. The sign and direction survive; the magnitude does not. At α=0 it
goes from −0.060 to −0.022.

**(b) "Chambers never recover at high α" is a T=200 artifact.** Exploratory
SECI at α=1 recovers from −0.41 to −0.16 (control) and −0.34 to −0.14 (main)
by T=600. The NHB draft's claim that chambers fail to recover for α ≥ 0.7
describes the length of the run, not the dynamics.

**(c) At true steady state, AECI-IE is ≈ 0 in every condition.** The
AI-channel echo signal lives entirely in the transient.

One effect *is* asymptotic and worth keeping: the exploiter L1+ pool recovers
to ≈ 20 at α=0 but stays at ≈ 6 at α=1 in both configurations. **Starvation
is permanent; the echo-chamber indices are not.**

## 7. What to do about it — author decision

The transient is not a nuisance here. A disaster response *is* a bounded
episode, and 200 ticks is a defensible response horizon; what the system does
by tick 600 may be irrelevant to a two-week emergency. But the paper must
then say so, and must stop calling that window a steady state.

- **(a) Re-run everything at T=600 and report the asymptote.** Honest and
  clean, but the effects shrink drastically, AECI-IE goes to zero, and most
  of the quantitative story dissolves. Cost: ~3× the compute of every sweep.
- **(b) Keep 200 ticks, reframe explicitly as a response horizon.** Replace
  "steady state, last 75 ticks" with a stated finite-horizon analysis, and
  make the primary evidence the trajectory features the lifecycle analysis
  already computes — peak depth, recovery tick, fraction of horizon spent
  in-chamber. These are well defined whether or not the system has settled.
- **(c) Both (recommended).** Lead with the response-horizon analysis, and
  report the T=600 asymptote in the SI as a robustness statement: *the
  chambers a confirming AI produces do eventually dissolve, but not within
  the response horizon, while the information starvation it causes is
  permanent.* That is a stronger and more defensible claim than the current
  one, and it costs one extra sweep rather than a re-run of all of them.

Whichever route, the phrase "steady state" should not survive into the
manuscript for these series, and the T=200 window means currently in the
draft should be relabelled as end-of-horizon values.
