# SECI/AECI co-evolution and the social→AI bubble handover

Analysis of the author's hypothesis: *at low α, explorers are AI-reliant
and thereby form a (shared-source) bubble; exploiters do the opposite —
they migrate from a social bubble to an AI-induced bubble as α rises.*
Plus the requested double-check of the exploit AECI-IE ≈ −0.03 value at
α=1.0 in the main configuration.

Data: high-α zoom on the main configuration (mobility=1, spatial_bridged,
network scope), α ∈ {0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}, 4 seeds,
150 ticks, late-run means; cross-checked against the N=20 × 200-tick CI
sweep (run 32404133354). Metrics: SECI; AECI-IE (served pool vs global
beliefs); AECI-IE-rel (served pool vs the community's OWN beliefs — new);
AI reliance (AI share of all delivered external reports — new);
effective α (delivered confirmation fraction — new).

## 1. The high-α "jump" is a rounding-saturation transition, not noise

Exploit AECI-IE across the zoom: −0.49 (α=0) → −0.54/−0.55 (0.3–0.6) →
−0.48 (0.7) → **−0.30 (0.8) → −0.17 (0.9) → −0.14 (1.0)**. What looked
like a jump from −0.45 (α=0.7) to −0.03 (α=1.0) in the 3-seed run is a
transition spread over α = 0.7→0.9 and flat from 0.9 to 1.0. Its location
is fully explained by the integer rounding of served reports
(`round(sensed + α·(belief−sensed))`): a 1-level belief–truth gap flips
from 0% to 100% delivered confirmation between α=0.5 and 0.6, 2-level
gaps at 0.75, and by **α=0.9 every gap size serves the caller's prior
exactly — α=0.9 and α=1.0 are literally identical at the report level.**
The new `effective_alpha` series makes this staircase measurable in every
run (nominal α on the x-axis understates confirmation in the mid-range
and saturates at 0.9).

Why the index goes toward 0 there: once the AI is a pure mirror, the
exploit community's AI-served pool is a sample of its own belief pool, so
AECI-IE inherits the social state instead of measuring an AI contribution.
Direct evidence: the per-seed correlation between exploit SECI and exploit
AECI-IE rises from r ≈ 0.1–0.6 (α ≤ 0.5) to **r = 0.95/0.91/0.89 at
α = 0.8/0.9/1.0** — at high α the AI-channel index is just re-measuring
SECI, seed by seed. The 4-seed spread at α=1 ({−0.11, −0.01, +0.04,
−0.47}) tracks each seed's SECI ({−0.32, −0.31, −0.02, −0.58}); the
earlier −0.03 was a 2-seed draw from this wide, SECI-slaved distribution.

Verdict: the value is real, mechanistic, and *not interpretable as "no AI
bubble"* — at α ≥ 0.9 the belief-baseline AECI-IE stops being an
independent measurement for exploiters. Model-design option if a smooth α
axis is wanted: probabilistic rounding of served reports (floor/ceil with
probability equal to the fractional part) makes E[report] linear in α and
removes the staircase; this is a trajectory-changing model edit and needs
an author decision.

## 2. Co-evolution: the handover hypothesis is supported for exploiters

Late-run means, main configuration (zoom, 4 seeds):

| α | SECI_ex | AECI-IE_ex | AECI-IE-rel_ex | reliance_ex | MAE_ex | SECI_er | AECI-IE_er | AECI-IE-rel_er | reliance_er | MAE_er |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | **−0.53** | −0.49 | −0.43 | 0.70 | 1.90 | −0.00 | −0.24 | −0.32 | 0.51 | **0.73** |
| 0.3 | −0.41 | −0.56 | −0.51 | 0.70 | 1.91 | −0.03 | −0.27 | −0.35 | 0.52 | 0.84 |
| 0.5 | −0.39 | −0.55 | −0.35 | 0.69 | 1.97 | −0.06 | −0.31 | −0.32 | 0.55 | 1.08 |
| 0.6 | −0.36 | −0.54 | −0.38 | 0.67 | 1.98 | −0.10 | −0.24 | −0.27 | 0.54 | 1.22 |
| 0.7 | −0.34 | −0.48 | −0.36 | 0.71 | 1.93 | −0.12 | −0.25 | −0.27 | 0.55 | 1.25 |
| 0.8 | −0.27 | −0.30 | −0.18 | 0.70 | 2.00 | −0.26 | −0.36 | −0.32 | 0.50 | 1.60 |
| 0.9 | −0.30 | −0.17 | **−0.04** | 0.73 | 1.92 | −0.36 | −0.49 | −0.60 | 0.52 | 1.79 |
| 1.0 | −0.31 | −0.14 | **−0.02** | 0.69 | 1.93 | **−0.36** | **−0.60** | **−0.68** | 0.52 | **1.82** |

**Exploiters — social → AI handover, visible in three co-moving series:**

1. **SECI shallows** from −0.53 (truthful AI) to ≈ −0.28…−0.31 (α ≥ 0.8):
   the *social* echo chamber weakens as α rises.
2. **AECI-IE-rel rises to ≈ 0**: at low-mid α the AI still injects
   something different from the community's own beliefs (rel −0.35…−0.51);
   by α ≥ 0.9 rel ≈ −0.02…−0.04 — the AI serves the community exactly its
   own beliefs back. **rel → 0 is the signature of the pure
   individualized echo**; the AI has stopped being an independent
   information source.
3. **Reliance is high throughout (≈ 0.70)**: ~70% of all reports
   exploiters receive come from the AI at *every* α. So the handover is
   not "they start using the AI more" (delivered-report share is flat;
   note the companion result from the earlier session that *query* share
   does rise, 0.58 → 0.69 at α=1) — it is that the same dominant channel
   flips from truth-anchored to mirror.
4. **MAE_ex stays ≈ 1.9 at all α**: the bubble's *content* (wrong beliefs)
   is unchanged — what changes is the mechanism that maintains it. At low
   α the social network sustains it against a truthful AI; at high α the
   individualized AI echo sustains it while the social chamber relaxes.

That is the author's hypothesis, with one refinement: the belief-baseline
AECI-IE cannot show "the AI-induced bubble appears" for exploiters
(because it collapses onto SECI exactly when the bubble becomes pure
echo); the **rel → 0 trajectory plus flat-high reliance plus persistent
MAE** is the correct quantitative signature of the handover.

**Explorers — reliant at every α; the bubble arrives via content, not
reliance.** Explorers get ~50% of their reports from the AI at all α, and
at α=0 this reliance is *beneficial*: MAE 0.73, SECI ≈ 0, mild IE indices.
The shared truthful source produces convergence toward truth, which the
IE constructs deliberately do not count as a bubble — so the "AI-reliant
therefore bubble" half of the hypothesis is not supported at α=0 by the
echo indices (if diversity-loss per se is the concern, the Shannon
info-diversity series is the right instrument). The explorer bubble
appears with α: AECI-IE deepens −0.24 → −0.60, rel deepens to −0.68 (the
AI serves them far less diversity than their own belief spread), SECI
deepens 0 → −0.36, and MAE rises 0.73 → 1.82 (+150%). The accuracy cost
of alignment falls almost entirely on the accuracy-seeking type
(Finding 2), and their social chambers *deepen* in step — the AI-channel
narrowing propagates into the social layer for explorers, the mirror
image of the exploiter handover.

## 3. Normalization — recommendation

"Should we normalize for the different communities?" Yes — that is what
AECI-IE-rel does (community served pool vs that community's own belief
variance), and it resolves both exploiter confounds: query concentration
cancels (both pools live on the community's chosen support) and the α=1
mirror maps to a fixed, interpretable value (0) instead of inheriting
SECI. The cost is that rel is *not* the same construct as SECI (it
measures the channel's marginal narrowing/broadening relative to the
receiver, not within-boundary homogeneity vs global), so
`total_bubble = |SECI| + |AECI-IE|` keeps the belief-baseline AECI-IE if
construct symmetry is wanted for the composite.

Proposed reporting scheme for the paper:
- **SECI** — social-layer echo (per type), as before.
- **AECI-IE (belief baseline)** — AI-channel narrowing vs global beliefs
  (per type); interpretable for explorers across the full α range and for
  exploiters up to α ≈ 0.8, with the α ≥ 0.9 collapse-toward-SECI stated
  as a property, not a bug (r ≈ 0.9 with SECI shown in SI).
- **AECI-IE-rel** — the mirror index; its → 0 trajectory carries the
  exploiter handover claim.
- **AI reliance + effective α** — context panels: channel mix of the
  information diet, and delivered (vs nominal) confirmation.

All four are in every output table/figure and in the re-dispatched N=20
sweep, so the choice of which to headline is editorial, not
computational.

## 4. Status

- Zoom data: `scratchpad` (session-local); summary figure
  coevolution_zoom.png shared in-conversation. Paper-grade co-evolution
  figures come from `tools/coevolution.py` applied to the N=20
  `experiment_results.json` artifacts (re-dispatched run from this
  branch, which carries rel/reliance/effective-α columns).
- Known measurement caveat: reliance counts *delivered reports* (AI
  answers ~25 cells/query; humans answer only cells they hold beliefs
  about), so it overweights the AI channel relative to a query-count
  share; both views are available (`mode_choice` data gives query
  shares).
