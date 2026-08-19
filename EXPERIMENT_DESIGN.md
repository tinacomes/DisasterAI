# Networked estimation experiment: belief-aligned AI in bounded vs. unrestricted groups

Design document for Phase N2 of `SUBMISSION_PLAN_NHB.md`. Becker et al.
(2017, PNAS)-style networked estimation, extended with (i) an AI advisor
whose confirmation level mirrors the model's alignment parameter, (ii)
endogenous source choice mirroring the model's Q-learning source selection,
and (iii) an on-site/remote role split mirroring the model's periphery
finding.

## 1. Research questions and model predictions under test

The experiment tests the three predictions of the agent-based model that do
not depend on its stylised disaster environment:

- **P1 (structural precondition, Finding 1).** A belief-confirming AI
  advisor deepens within-neighbourhood belief convergence *and* error only
  when communication is network-bounded; under unrestricted visibility it
  degrades accuracy without deepening local convergence.
- **P2 (truthful-AI bubble, Finding 2).** A truthful AI advisor produces
  convergence among its heaviest users (high advice-reliance participants)
  in *both* structures — reliance-defined convergence, not tie-defined.
- **P3 (periphery, Finding 4).** The harm of a confirming advisor
  concentrates on participants without direct evidence access ("remote"
  role), and their contribution/engagement declines; network position does
  not protect them.

A secondary, mechanism-level prediction from the query-share result: over
trials, a confirming advisor captures a growing share of participants'
information choices, while a truthful advisor's share stays flat.

## 2. Task

Estimation with objective ground truth, disaster-framed: participants view
post-disaster aerial/satellite images of neighbourhoods and estimate the
**share of buildings destroyed or severely damaged (0–100%)**. Stimuli and
ground truth from the public **xBD/xView2 building-damage dataset**
(expert-annotated damage polygons; select ~15 images spanning 5–85% true
damage, pre-piloted for difficulty). Disaster framing preserves the paper's
setting; the quantity is objective, continuous, and incentive-compatible.

Fallback neutral task (if imagery proves too noisy in piloting): standard
numeric estimation (e.g., objects in an image), losing the disaster frame
but keeping all treatments intact.

## 3. Groups, network, and roles

- **Group size**: 16 participants, interacting synchronously.
- **Networked structure**: two "communities" of 8, each a ring lattice with
  degree 4, connected by 2 bridge ties — a minimal analogue of the model's
  community-plus-weak-ties graph. Participants see only their neighbours'
  estimates.
- **Unrestricted structure**: each round, every participant sees the
  estimates of 4 group members drawn at random from the whole group
  (matched information volume; the analogue of "anyone can query anyone").
- **Roles (within every group)**: 8 **on-site** participants see the image;
  8 **remote** participants never see the image and must rely on peers and
  the advisor. Roles are assigned so each community/neighbourhood contains
  both. Remote participants earn the same accuracy bonus, so withdrawal is
  measurable as disengagement, not payoff-rational inaction.

## 4. Treatments (2 x 2 between groups, + baseline)

| Factor | Levels |
|---|---|
| Communication structure | networked vs. unrestricted |
| AI advisor policy | truthful vs. confirming |

- **Truthful advisor**: advice = ground truth + zero-mean noise (s.d.
  calibrated in piloting to match the model's AI sensing noise; identical
  noise draw logic across arms).
- **Confirming advisor**: advice = (1 − α)·(truth + noise) + α·(mean of the
  participant's *visible neighbourhood's* previous-round estimates), with
  α = 0.8. Targeting the local consensus rather than the participant's own
  prior mirrors the model's trusted-network-consensus mechanism — the
  ingredient the ablation (PNAS plan, Phase 1) tests in silico. If budget
  allows a 5th arm: confirming-own-prior (b = participant's own last
  estimate), the experimental analogue of that ablation.
- **Baseline arm (no advisor)**: networked structure without AI, to anchor
  the social dynamics against Becker's published results.

The advisor is presented as "an AI damage-assessment assistant". This is
accurate (it is an algorithm); that its advice is sometimes
accuracy-degraded is disclosed in debriefing (see Ethics).

## 5. Trial structure and source choice

Per group: **10 trials** (images), each with three rounds in the Becker
tradition, ~25–30 minutes total:

1. **R1 — private**: initial estimate + confidence (0–100 slider).
2. **R2 — informed revision**: participant *chooses one source to consult*
   — "peers" (visible neighbours'/draws' R1 estimates) or "AI assistant"
   (advice computed as per treatment) — then revises estimate + confidence.
3. **R3 — final**: same choice again (may switch source); final estimate +
   confidence. Final estimate is the incentivised one.

The forced single-source choice per round is the design's distinctive
element: it operationalises the model's Q-learning source selection and
yields the **AI query share** trajectory (secondary prediction). A pushed
variant (both sources always shown, as in classic Becker) is kept as a
pilot comparison in case choice makes the task too heavy.

**Individual differences**: 3-minute post-task battery — Actively
Open-minded Thinking short scale + a brief need-for-closure measure — as a
proxy for the model's exploratory/exploitative types, used in secondary
moderation analyses only.

## 6. Outcome measures (mapped to model metrics)

| Model metric | Experimental analogue |
|---|---|
| SECI | Within-neighbourhood variance of final estimates relative to whole-group (and cross-group population) variance, per trial |
| AECI-Var | Same variance contrast, grouping by advice-reliance (median split on weight-on-advice) instead of by neighbourhood |
| AECI-Err | Confidence-weighted absolute error contrast, advice-heavy vs. advice-light participants |
| Belief MAE | |final estimate − ground truth|, per participant-trial |
| AI query share | Share of R2/R3 source choices going to the advisor, per participant over trials |
| Periphery gap | Remote-minus-on-site differences in error, confidence, and revision engagement |
| Relief withdrawal | Remote participants' engagement decline (response latency ceiling-outs, non-revision) across trials |

Weight on advice (WOA) computed per revision in the standard way from
estimate movement toward the consulted source.

## 7. Hypotheses (preregistered), analysis

- **H1** (P1): structure x advisor interaction on within-neighbourhood
  convergence and on final MAE — confirming advisor increases convergence
  and error in networked groups more than in unrestricted groups.
- **H2** (P2): under the truthful advisor, advice-reliance-grouped
  convergence (AECI-Var analogue) exceeds neighbourhood-grouped
  convergence, in both structures.
- **H3** (P3): the error and disengagement cost of the confirming advisor
  is larger for remote than on-site participants (role x advisor
  interaction), and is not moderated by network position (degree/bridge
  status) — a predicted null, tested with equivalence bounds.
- **H4** (secondary): AI query share rises over trials under the confirming
  advisor and stays flat under the truthful one.

Mixed-effects models with group and participant random effects; trial as
random effect; group is the unit of randomisation. Primary tests
Holm-corrected. **Power analysis is simulation-based from the ABM itself**:
run the model at matched parameters (16 agents, the experimental graph, 10
"trials", α ∈ {0, 0.8}) to obtain expected effect sizes for H1, then
simulate the mixed model. Ballpark until that is run: 10 groups per cell
(2x2 + baseline = 50 groups, 800 participants; with 25% oversampling for
synchronous attrition ≈ 1,000 recruits). If simulated effects are large,
8 groups/cell (~640+attrition) may suffice — decide after the power
simulation, not before.

## 8. Platform, recruitment, incentives, budget

- **Software**: Empirica (first choice for synchronous networked
  experiments) or oTree; bot-fillable seats to rescue groups from single
  dropouts; waiting-room lobby with attention check.
- **Recruitment**: Prolific, English-fluent, desktop only; synchronous
  sessions scheduled in blocks.
- **Pay**: base ≈ £4.00 (25–30 min at ≥£9/hr equivalent) + accuracy bonus
  up to £1.50 (linear in negative mean absolute error over trials) + £0.50
  completion-of-group bonus (attrition control).
- **Budget envelope** (1,000 recruits x ~£6 average incl. platform fees):
  **≈ £6,000–8,000**, plus piloting (~£500).

## 9. Ethics and open science

- TU Delft HREC application: minimal-risk behavioural study; the single
  sensitive point is **partial deception** — the confirming advisor is
  described as an AI assistant without disclosing that its advice blends in
  the group's own prior estimates. Mitigation: full algorithmic disclosure
  in debriefing, option to withdraw data post-debrief. Disaster imagery is
  of buildings, not people; screen stimuli to exclude visible casualties.
- Preregistration on OSF (hypotheses H1–H4, exclusion rules, models) after
  piloting, before main data collection.
- Data and analysis code to the same Zenodo archive as the model.

## 10. Timeline

| Weeks | Activity |
|---|---|
| 1–5 | Software build (Claude Code), stimulus selection from xBD, internal testing |
| 1–8 (parallel) | HREC application and approval |
| 6–7 | Pilot (2–3 groups incl. one pushed-information group), fix task difficulty and advisor noise; ABM-based power simulation |
| 8 | Preregistration |
| 9–12 | Main data collection |
| 13–16 | Analysis, model–experiment docking figure, manuscript integration |

## 11. Claude Code build prompt

```
In tinacomes/DisasterAI, read EXPERIMENT_DESIGN.md and implement the
experiment in a new top-level directory experiment/ using Empirica
(fallback: oTree if Empirica tooling is unavailable): 16-player synchronous
games, the 2x2+baseline treatments, the two-community ring-lattice-with-
bridges graph and the random-draw unrestricted condition, on-site/remote
roles with image visibility gating, the R1-R2-R3 trial structure with
forced single-source choice, advisor logic per Section 4 (alpha=0.8
confirming advisor targeting the visible neighbourhood's previous-round
mean), confidence sliders, WOA-ready data export matching Section 6, a
bot-player smoke test that runs a full 16-seat game headlessly, and a
config file exposing group size, alpha, noise s.d., trial count, and
stimulus list. Separately add experiment/power_simulation.py: run the
existing ABM (DisasterAI_Model.py) at matched parameters to extract H1
effect sizes and simulate the mixed-model power curve over groups-per-cell
in {6,8,10,12}. Do not collect or hardcode any real stimuli; use
placeholder images and document where xBD selections plug in.
```
