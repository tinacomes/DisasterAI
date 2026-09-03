# Paper brief — belief-aligned AI in networked disaster response (DisasterAI)

This is the master briefing for redrafting the paper **from scratch**. It
supersedes `PNAS_WRITING_INSTRUCTIONS.md` as the drafting entry point (that
file remains valid as PNAS-format history and detailed panel specs, but the
venue decision below replaces its §1). Everything a writer needs is in this
repository: `RESULTS_COMPENDIUM.md` (every claim → dataset/figure/table),
`RESULTS_OVERVIEW.md` §0, `METHODS_PAPER.md`, and the per-directory result
READMEs. **Do not re-derive numbers — cite them from the compendium**, whose
headline claims (H1–H10) this brief reuses by ID.

**Status: the results are frozen and the validation program is complete.**
Canonical dataset: **`results-mechfix/`** (run 32821105202, commit
`30f89e0`), with the ablation, counterfactual, boundary, robustness,
docking, and dynamics chains all run and passed (compendium §4). The
existing `PNAS_Paper/` draft is **quarry material** (figures,
`References.bib`, TikZ concept figure), not the base text: the redraft
starts from this brief.

## How to use this pack

1. Read this brief end to end, then `RESULTS_COMPENDIUM.md` (headline
   claims H1–H10 with their evidence paths), then `METHODS_PAPER.md`
   (authoritative model description). `PNAS_Paper/STORYLINE.md` records
   which claims from the old NHB storyline are dead and why — read it once
   so none of them creep back in. Cite from `PNAS_Paper/References.bib`
   only; do not introduce citations from memory; resolve the
   `steinbrink2024transparency` placeholder before submission or leave it
   uncited.
2. Draft from the skeleton at the bottom. Every claim you write must trace
   to a compendium H-item or a named results table; the framing and
   phrasing rules below are binding — they encode what the model and the
   ablation chain can and cannot support.
3. Background only if needed: `docs/robustness_summary.md` (M5 verdicts),
   `results-*/README.md` (per-run provenance and verdict paragraphs),
   `RESULTS_OVERVIEW.md` §§1–8 (the historical chain — superseded numbers,
   SI transparency material only).

## Target venue (APC-constrained; revised 2026-09-03)

The paper is theory- and model-based with no empirical data of its own;
its external-validity anchor is the docking chain (M6 qualitative
reproduction, compendium §4, plus the E1 effect-size fit in
`results-docking-fit/`). The emphasis is AI–human interaction at
population scale, and the author's goals are **readership and
discussion**.

**Binding constraint (author, 2026-09-03):** library support at TU Delft
and at DLR covers at most ≈ USD 2,000 per article and nothing beyond.
That removes every gold-OA-only route: Science Advances (USD 5,450),
Nature Communications (USD 7,350), the Nature Human Behaviour OA option
(USD 12,850) and PNAS Nexus. The Dutch Springer Nature agreement and
DEAL both exclude Nature-branded journals. The strategy is therefore
zero-APC: subscription routes, journals covered by the Dutch Elsevier /
Wiley agreements (TU Delft corresponding author; valid to 31 Dec 2026)
or by DEAL (DLR corresponding author; 2024–2028), or Subscribe-to-Open
journals — with the arXiv preprint at submission and, on the
subscription route, the version of record in the TU Delft repository
six months after publication under the Taverne amendment ("You share,
we take care"). Cost table and verification notes:
`docs/development/EMPIRICAL_REVIEW_2026-09-03.md` §4.

Still excluded by author decision: PNAS/PNAS Nexus (PNAS's delayed-OA
route would be affordable at ≈ USD 2,575, but the exclusion stands);
JASSS and J. Computational Social Science (social-simulation audience,
not the AI + behaviour audience); Collective Intelligence and other
young non-indexed venues (prior submission experience). Nature Machine
Intelligence is free on its subscription route but remains the scope
stretch flagged before and is not recommended.

| Priority | Journal | Cost route | Fit | Main risk |
|---|---|---|---|---|
| 1 | **Nature Human Behaviour** (subscription route) | No APC; arXiv + Taverne green OA at 6 months | The docking target is an NHB paper; NHB's own agenda pieces (*Machine culture*, 2023; *A new sociology of humans and machines*, 2024) ask for human–machine social systems studied with computational models; ABM and human–AI interaction both at home | Highest bar; simulation-only research articles are rare there — presubmission enquiry first; desk rejection is the modal outcome and costs weeks, not months |
| 2 | **Computers in Human Behavior** (Elsevier, hybrid) | OA at no cost under the Dutch Elsevier agreement or DEAL | Scope is the psychological impact of computing on "individuals, groups and society", the computer "only as a medium through which human behaviours are shaped" — the brief's behaviour-first rule verbatim; ABM/simulation studies of echo chambers appear there; IF 12.2; no hard length cap | Psychology reviewers asking for human data — answered by the E1 fit, the E4 predictions table, and the Cheng et al. 2026 sycophancy→dependence anchor; long queues |
| 3 | **Journal of the Royal Society Interface** | Subscribe-to-Open 2026: OA, CC-BY, no APC (2027 continuation to verify) | Modelling audience with behavioural reach: ABMs of opinion dynamics with social-identity bias, collective wisdom under incorrect social information, misinformation on networks are in its recent record; the Lorenz/Becker lineage argument lands | Narrower AI readership (arXiv carries it); the acceptance date must fall inside the S2O year |

Fallbacks, all covered by the same agreements: International Journal of
Human-Computer Studies (Elsevier), Technological Forecasting & Social
Change (Elsevier), Royal Society Open Science (only if a Royal Society
Read & Publish covers TU Delft — verify in the library's Journal
Browser).

**Recommendation and decision rule:** redraft once, framed as human–AI
collective behaviour with the ABM as the instrument (framing rules
below, plus two venue-neutral additions from the review §4: lead with
people, never with alignment machinery; anchor the motivation in the
measured dyadic and sycophancy evidence — Glickman & Sharot 2025, Cheng
et al. 2026 — and state the population-scale gap in one sentence).
Presubmission enquiry to NHB at the end of the redraft; on a decline or
no reply within three weeks, submit to Computers in Human Behavior;
Interface if CHB rejects on "no human data". **Independent of venue:
post the preprint to arXiv (cs.CY + cs.MA, cross-list physics.soc-ph)
at submission** — the AI community discovers work through arXiv, not
journal tables of contents; this is how the paper reaches that audience
whichever journal carries it.

**E2 status (author decision 2026-08-31): parked as a standalone side
project** — a measurement study of its own, not part of this paper's
pre-submission path (full plan preserved in
`docs/ALPHA_MEASUREMENT_PROTOCOL.md`; rationale: it is a second study,
and the paper's length budget cannot carry a Results + Methods +
figure block for it). If the side project produces a citable preprint
in time, the Discussion cites it as companion evidence — that
strengthens the paper but does not change the venue call.

**What the venue change buys the redraft:** none of the three targets
imposes the PNAS corset (6-page limit, 4-display-item cap, Significance
Statement). CHB and Interface have no hard main-text cap: write
~5,000–7,000 words with 5–6 main figures and pull the credibility chain
(dose-linearisation ablation, docking + E1 fit, salience counterfactual)
*into the main text* instead of burying it in the SI — that compression
was the standing weakness of the PNAS draft. NHB compresses the same
skeleton to its Article length with Methods after Discussion; CHB wants
Introduction–Method–Results–Discussion, so Methods moves before Results
without other changes. Keep American English (NHB house style; CHB and
Interface accept either).

## The paper in one paragraph (abstract seed)

# writing note: the abstract must be readable by a non-modeler; name the
# mechanisms in plain words (echo chamber, confirmation, starvation,
# capture); few numbers, no acronyms; end on the governance implication.

People increasingly ask AI systems, rather than each other, what is
happening — nowhere more consequentially than in disasters, where time
pressure is extreme, stakes are existential, and verification is degraded.
Preference-trained AI drifts toward confirming user beliefs, and the
evidence on what that does stops at individual users, while echo chambers
live in networks. We develop an agent-based model of decentralized
disaster response in which a single parameter doses the AI's answers from
ground truth to full confirmation of the querier's own prior, and compare
seed-paired populations under network-bounded versus unrestricted
information access. Social network structure, not the AI policy alone,
decides the failure mode: unrestricted access dissolves the
confirmation-seekers' echo chamber even under a fully confirming AI, while
network-bounded access preserves it at every dose. The harm of
confirmation is starvation, not persuasion — accuracy-seekers bear the
entire accuracy cost because a confirming AI echoes their priors and
empties their belief pool — while confirmation-seekers are captured into
rising reliance without rising trust. Neither full truthfulness nor full
confirmation is operationally optimal: an interior alignment level
minimises unmet needs and maximises the diversity of the served
information environment, robustly across metric definitions and cognitive
profiles. Harms concentrate on the spatial periphery and outlive the
policy: repairing the AI dissolves chambers within weeks of model time,
but the epistemic starvation persists longer than the exposure that caused
it. The population-scale harm of sycophantic AI is the silent
reorganization of who learns what from whom; monitoring must target
community-level epistemic dynamics, not individual accuracy.

## Research questions

RQ1. Structural preconditions: under which access conditions (network
     topology, mobility, query scope) does belief-aligned AI amplify —
     versus dissolve — social echo chambers?
RQ2. Dose–response: how do epistemic and operational harms scale with the
     alignment dose α, and is there an interior optimum?
RQ3. Feedback and heterogeneity: who gets captured, who gets starved, and
     can informational interventions (making disconfirmation salient)
     break the loop?
RQ4. Distribution and dynamics: where do harms concentrate, and are they
     reversible once the AI policy is repaired?

## Contribution claims

1. A minimal, **type-agnostic** formalisation of AI belief-alignment as a
   dose — one response rule for all users, r = (1−α)·truth + α·prior,
   delivered dose linear in α — inside a networked, spatially embedded
   ABM of disaster response; extends the dyadic human–AI feedback-loop
   evidence (Glickman & Sharot) to collective outcomes in the
   Lorenz/Becker lineage of network epistemics.
2. **The structural-precondition result** (H3): network-bounded access is
   what lets a confirming AI preserve the confirmation-seekers' community
   echo chamber; under unrestricted access, out-group contact dissolves
   it. The failure mode is a property of the sociotechnical system.
3. **A mechanism pair replacing "AI persuades people":** *starvation* of
   accuracy-seekers (belief-pool collapse, the entire accuracy cost) and
   *capture* of confirmation-seekers (reliance rises, trust flat), with
   two boundary conditions — accuracy-seekers never learn to distrust a
   confirming AI because verification is base-rate dominated (C12), and
   making disconfirmation salient backfires into social retrenchment
   rather than truth-seeking (H4, H5, H6).
4. **An interior alignment optimum** α\* = 0.6 that is composite-robust
   (7/12 definitions), cognitive-profile-robust (all 132 sweep cells),
   ablation-attributed (dose linearisation, not the confirmation
   reference), and echoed by the societal layer: the served information
   environment is most diverse at intermediate alignment, coinciding with
   the operational optimum (H1, H2, H7).
5. **Distribution and dynamics of harm** (H8, H9, H10): harms concentrate
   on the spatial periphery (absent in the immobile control); chamber
   formation is universal but dissolution is dose-blocked and capture
   onset accelerates ~25×; repairing the policy dissolves chambers with a
   median ~50-tick lag, yet starvation outlives the policy and an
   informed population is buffered against late-onset confirmation — the
   harm is front-loaded into belief formation.
6. A transparent **attribution architecture** usable as a template:
   seed-paired configurations, single-mechanism ablations, N=50 boundary
   replication with Holm correction, ten-perturbation robustness envelope,
   dyadic docking against a published experiment, and a full transparency
   chain across model revisions.

## Evidence map (claim → key numbers → where)

IDs are `RESULTS_COMPENDIUM.md` §2 items — full file paths live there;
this table is the drafting shorthand. All numbers: `results-mechfix/`
unless the compendium says otherwise; steady state = last 75 of 200
ticks; N = 20 seed-paired replications (N = 50 for boundary claims).

| # | Claim | Key numbers | Evidence (compendium pointer) |
|---|---|---|---|
| H1 | Interior optimum α\*=0.6, operational coincidence | 7/12 composites at 0.6 (spread 0.1–0.6); unmet needs 1.59 → 0.25 (α=0.6) → 2.86 | `goldilocks_alignment_sweep.png`, `alpha_star_sensitivity.png`; `comparison.txt` |
| H2 | Optimum attributable to dose linearisation | detround ablation fragments α\* to [0.2, 0.6]; ownref ablation inert (8/12 at 0.6) | ablation `summary_table.md` ×2; mechfix README §Attribution |
| H3 | Bounded access = structural precondition | SECI_exploit −0.45 → **+0.05** (global) vs −0.39…−0.18 (bounded); ΔSECI_exploit at α=1 −0.269 [−0.350, −0.189], N=50, Holm p≈5e-08; explorer deepening 0 → −0.33 in both | `comparison_configs.png`; paired deltas in `comparison.txt`; `results-boundary-n50/` |
| H4 | Starvation, not persuasion | explorer MAE 0.54 → 1.74, exploiter flat ≈1.8; L1+ pool 113 → 29; smooth dose–response | MAE panels; `summary_table.csv` (`l1pool_*`) |
| H5 | Capture, contingent on base-rate-diluted evaluation | AI query share 0.54 → 0.69, trust flat ≈0.47–0.49; s=1 removes the gradient but chamber deepens (−0.52 vs −0.37) and precision falls (0.43 vs 0.57) at α=0 | `aeci_evolution.png`; `results-salience/` tables + verdict |
| H6 | C12: explorers never punish confirming AI via trust | trust 0.88 → 0.84 (s=0); 0.90 → 0.82 at s=1 | `robustness_tables.md`, `AItrust_er` |
| H7 | Fragmentation vs societal layer | SECI-pop −0.22 → 0.00 masks the per-type crossing; AECI-IE-chan-pop U-shaped, shallowest ≈ α=0.7; population composites most ablation-robust α\* (0.6 in all three datasets) | `population_evolution.png`; composite rows in `comparison.txt` |
| H8 | Operational buffering + spatial periphery | unmet needs 2.86 vs 10.18 at α=1; explorer precision 0.62 vs 0.20; MAE gap +0.12 → +0.33, aid gap −1.8 → −6.6 (≈0 in immobile control); betweenness/broker gaps small | `periphery_gap.png` (+ evolution); paired deltas in `comparison.txt` |
| H9 | Hysteresis: repair works, starvation outlives it, history protects | dissolution 18/19 (from α=1, median lag 50 ticks) vs 4/19 constant; post-repair pool 102 vs 120, MAE 0.49 vs 0.41 (CI-separated); late-onset buffered (MAE 0.57 vs 1.59) | `results-reversal/` trajectories + summary + verdict |
| H10 | Lifecycle grid: formation universal, dissolution dose-blocked, capture onset ~25× faster | formation 20/20 everywhere (tick 22±1 → 67±7); dissolution 17/20 → 3/20 (main); capture onset 59±3 → 2±1 | `results-lifecycle/lifecycle_perseed.md` (per-seed RNG-drift caveat; endpoints cite mechfix) |
| — | Docking (external-validity anchor) | Glickman–Sharot dyadic amplification reproduced for both types; aligned AI retains more bias than a human partner | `results-docking/` (M6); effect-size fit `results-docking-fit/` (E1, population sweep pending) |
| — | Robustness envelope | all criteria hold: population 100–500, small-world generator, AI supply 1–10, verification 0.1–0.5; 132-cell cognitive-profile sweep interior everywhere | `results-robustness/`, `docs/robustness_summary.md`, `results-gap-sweep-mechfix/` |

## The model's own robustness (the referee's first question)

The paper is a simulation with no field data, so the attack is "your α is
a cartoon and your metrics are constructed." The answer has four layers,
and the redraft should state them **in the main text**, not the SI:

1. **The dose is real and attributable.** Stochastic report rounding makes
   the *delivered* confirmation dose linear in α (manipulation check:
   `effective_alpha`; cite once). The single-mechanism ablations attribute
   the robust interior optimum to exactly this linearisation, while the
   confirmation-reference revision is behaviourally inert — the model's
   headline behaviour does not hinge on a hidden modelling choice (H2).
2. **The metrics are honest about what they can see.** AECI-Var was
   retired *by the project itself* because it conflates convergence-on-truth
   with an echo chamber and is blind to the individualized bubble; the
   AI-side construct (AECI-IE-chan) applies SECI's own variance-ratio
   formula to the served information, is ≈0-diverse at the truthful
   endpoint, and is reported per type and at population level. Telling
   this retirement story briefly is a credibility asset, not a confession.
3. **The dyad docks to a published experiment.** One human × one AI, no
   network, no relief: the model qualitatively reproduces Glickman &
   Sharot's human–AI bias amplification for both cognitive types, and
   the E1 fit estimates the dyad's micro-parameters from their published
   effect sizes (SI; stated with its conservative mismatch). The
   collective results are then the model's *addition* to a validated
   dyadic core, in the same move Lorenz/Becker made for social influence.
4. **Nothing rests on a knife edge.** Every high-α boundary claim is
   seed-robust at N=50 with Holm correction; the robustness envelope holds
   at every perturbation level (population, topology generator, AI supply,
   verification probability); the interior optimum appears in every cell
   of the 132-cell cognitive-profile sweep; and a full transparency chain
   (legacy → type-agnostic fix → mechanism revision) is tabled in the SI.

## Empirical foundations — current state and upgrade path

**Current state.** The model's empirical touchpoints are: (i) the
qualitative docking against Glickman & Sharot — explicitly pattern
reproduction, *not* parameter-matched calibration (SI §M6 as archived;
the E1 fit below upgrades it once its population sweep is run and the
section is folded in); (ii) two
empirically sourced entries in the calibration table (returner–explorer
mobility, Pappalardo et al.; weak-tie structure, Granovetter); (iii)
design-rationale justifications for everything else. The steps below are
ordered by cost. E1–E4 are pre-submission work for THIS paper; E5–E6
strengthen it if time allows; E7–E8 are the follow-up programme and enter
the paper only as the stated research agenda in the Discussion — the
paper is not held for them. **Concrete Tier-1 implementation plan
(verification, E1/E3/E4 work packages, execution order, dated venue
re-confirmation): `docs/EMPIRICAL_FOUNDATIONS_PLAN.md`.**

**Tier 1 — pre-submission (weeks, no new human data):**

- **E1 — Effect-size-targeted docking — FIT EXECUTED, DELIVERABLE
  PENDING** (fit 2026-08-31, reviewed 2026-09-03; was
  "parameter-matched", rescoped because G&S's tasks share no units with
  the severity grid). Dyadic acceptance/trust parameters fitted to
  transmission coefficients computed from Glickman & Sharot's published
  primary data (pipeline `experiments/docking_fit/`, results and
  provenance `results-docking-fit/`). Honest reading of the fit: the
  human–human target is uninformative (its CI spans zero); the model
  under-transmits AI influence for both types (below the measured CI,
  ≈0 for confirmation-seekers by construction of the D/δ window, cf.
  C12 — a limitation to state, not a finding); the fitted point sits on
  the certified search-box boundary in two coordinates; the one
  identified direction is the initial AI trust (≈2× default).
  Population structure is verified intact on a reduced grid only.
  Still to do before "micro-parameters empirically estimated;
  population results unchanged" can be written: rerun the fit with the
  Exp2 accurate-AI error ratio as a second target, then the canonical
  N=20 sweep at the fitted set (workflow `run-docking-fit-sweep.yml`,
  archive `results-docking-fit-sweep/`), then the SI §M6 fold-in. Open
  items: `docs/development/EMPIRICAL_REVIEW_2026-09-03.md` §2.
- **E2 — PARKED as a standalone side project** (author decision
  2026-08-31): measuring the confirmation weight of deployed
  assistants with the model's own estimand is a study in its own
  right and does not fit this paper's length budget. Full plan:
  `docs/ALPHA_MEASUREMENT_PROTOCOL.md`; it proceeds on its own track
  toward its own paper. In THIS paper it appears only as one
  Discussion sentence in the research agenda (alongside E7/E8), or as
  a companion citation if a preprint exists by submission.
- **E2-lite — literature-anchored α plausibility (replaces E2 in this
  paper).** One Discussion paragraph anchoring the simulated α range
  in published measurements of assistant sycophancy: deployed systems
  demonstrably sit strictly between the truthful and fully confirming
  endpoints, and published pushback/multi-turn results show the
  confirmation weight rising under user pressure — so the interior of
  the dose axis, where the paper's mechanisms live, is the
  empirically relevant region. Strictly qualitative: no α̂ values
  derived from others' benchmarks (their estimands differ); citations
  verified into References.bib. Buys most of E2's rhetorical work for
  ~120 words and zero new data.
- **E3 — Calibration table upgrade.** For each behavioural parameter in
  Table S7, cite the empirical literature that estimates it (bounded-
  confidence window estimates; human–automation trust-updating rates;
  verification lags from situation-report cycle times in the crisis
  coordination literature). Changes no result; changes reviewer priors.
- **E4 — Predictions table.** One Discussion table of falsifiable,
  measurable signatures: accuracy costs concentrated on accuracy-
  seekers; reliance rising while trust is flat (visible in usage logs);
  query-diversity narrowing (visible in chat logs); repair lag and
  front-loaded harm. Converts "just a simulation" into a hypothesis
  generator.

**Tier 2 — if time allows (months, existing data):**

- **E5 — Stylised-fact validation** against crisis-informatics corpora
  (CrisisLex-type datasets): community belief homogeneity, correction
  dynamics, periphery information gaps — pattern-oriented validation,
  not calibration.
- **E6 — Empirically embedded structure.** Re-run the main configuration
  on an observed disaster-communication network and CDR-fitted mobility
  (the Pappalardo split comes with fitted parameters). Likely a
  confirmation (M5 already passes a topology swap), but "holds on an
  empirical network" reads differently.

**Tier 3 — follow-up programme (6–18 months; separate papers / grant
work packages, cited in the Discussion as the agenda):**

- **E7 — Networked human experiment** in the Lorenz/Becker tradition:
  participants estimate severity in a map task on Empirica/oTree
  networks (bounded vs. global visibility), querying a system-prompted
  LLM whose confirmation level is the manipulated treatment. Directly
  tests starvation, capture, and the structural precondition. Ethics
  approval + ~€10–20k participant costs.
- **E8 — LLM-agent replication**: re-run the sweep with LLM-driven
  agents (or the AI side played by real LLMs at their E2-measured α) to
  show the mechanisms are not artifacts of hand-coded rules; the
  LLM-population studies (e.g. Sci. Adv. 2025) are the genre precedent.

**Venue coupling:** with E2 parked, the paper stays simulation-only
(anchored by the docking chain); the venue call is the zero-APC tier
list in the venue section (NHB subscription route → Computers in Human
Behavior → J. R. Soc. Interface), with no contingency on E2.

## Framing rules (binding)

- **Behaviour first, model as instrument.** The message is human–AI
  behaviour at scale — starvation, capture, retrenchment, structural
  preconditions. The ABM is the instrument that makes these mechanisms
  measurable; never lead a section with model machinery.
- **One AI, one rule.** The AI is type-agnostic: r = (1−α)t + αb with
  b = the querier's own current belief, for every caller. Everything
  differential between agent types is *human* (acceptance windows,
  verification, rewards). A smoke test enforces this
  (`test_confirmation_target.py`); mechanism sentences must name the
  agent-side process.
- **Per-type reporting is the default** for SECI, MAE, trust, query
  shares — with the population layer reported alongside, because the
  societal lens *masks* the per-type crossing (H7). Combined indices never
  conceal a per-type divergence.
- **Sign convention stated once:** negative SECI/AECI = echo chamber;
  every index named with its construct on first use.
- **Cross-configuration claims use raw paired per-seed deltas.** Never
  compare within-sweep range-normalised composites across configurations.
- **α\* is always fully dressed:** 0.6, with the 7/12 composite count, the
  [0.1, 0.6] sensitivity spread, and the ablation attribution. Never a
  naked point value.
- **The optimum is descriptive, not a design target.** The Discussion
  carries the explicit disclaimer plus adoption routes that need no
  confirmation (uncertainty communication, transparency, verification
  support).
- **Extreme-case logic stated once, early:** disasters concentrate time
  pressure, stakes, and degraded verification — the sharpest, most
  measurable form of the mechanism; misallocation translates directly
  into unmet needs. Transfer to milder settings is by that logic, not by
  point prediction.
- **Theory-based positioning:** mechanisms and structural conditions are
  the transferable claims; point values (α\*, gap magnitudes) are
  model-relative. The docking result is the external-validity anchor and
  is cited wherever transfer is implied.
- **Truth-convergence disambiguation:** convergence at the truthful
  endpoint is not an echo chamber; keep the disambiguation paragraph and
  read AI-side indices together with error metrics.
- **Starvation context travels with SECI:** part of the explorer
  deepening is belief-pool shrinkage, not convergence on shared error —
  said explicitly once, and the L1+ pool is reported alongside SECI.
- **Statics and dynamics cite different datasets:** endpoint/steady-state
  numbers cite `results-mechfix/`; lifecycle and reversal claims cite
  `results-reversal/` (CI-grade) and `results-lifecycle/` (per-seed
  grid, with its RNG-drift caveat).

## Phrasing rules / prohibited claims

1. Never "consensus recycling" / "the AI echoes the community's beliefs
   back" — that mechanism described legacy dead code and the ablation
   shows the targeting is inert. The AI confirms the *querier's own
   prior*.
2. Never describe the AI as treating agent types differently — anywhere,
   including figure captions.
3. AECI-Var is never evidence for or against an AI bubble (retired;
   α\*-sensitivity table only). The AI-side echo construct is
   AECI-IE-chan (channel baseline), per type + population.
4. Never present SECI deepening without the belief-pool context (H4).
5. No main-text numbers from `results/`, `results-verification/`, or
   `results-final/` — superseded; SI transparency chain only.
6. The old control-configuration corner solution (α\*=1.0) is a
   dose-delivery artifact, never a model property.
7. The salience counterfactual is a boundary condition, never a remedy:
   at s=1 capture disappears via *social retrenchment* (deeper chamber,
   worse precision at the truthful endpoint), not truth-seeking.
8. Never "explorers learn to distrust confirming AI" — they don't (C12);
   base-rate-dominated verification is itself a finding.
9. The cognitive-gap sweep supports a *robustness* claim (interior
   optimum in every cell), never a profile-dependence claim; the per-type
   drift is suggestive-only with CIs.
10. Never frame the interior optimum as a recommendation to build mildly
    confirming AI.
11. "Seed-robust" only when the paired per-seed CI excludes zero; report
    steady-state means with SE or 95% CI; two significant decimals for
    indices.
12. No overclaimed external validity: reduced-form α is not any deployed
    system; no "prove", no "optimal policy" language.
13. No figuresplaining ("Fig. 2 shows that…"); the claim is the
    paragraph's first sentence, present tense, one claim per paragraph.
14. No references beyond `PNAS_Paper/References.bib`; do not resurrect
    the removed Fu et al. 2023 claim; resolve or drop the
    `steinbrink2024transparency` placeholder.

## Methods facts the writer needs (full text: METHODS_PAPER.md)

30 × 30 grid disaster environment; 100 human agents, 5 AI agents sensing
15% of the grid per tick (constant noise, independent of α). Two cognitive
styles differing only in *human* machinery — exploitative
(confirmation-seeking; narrow, sharp acceptance D=2.0/δ=3.5; slow trust
learning; scores confirmation against the trusted-network consensus when
defined, own prior otherwise) and exploratory (accuracy-seeking; wide,
gradual acceptance D=4.0/δ=1.2; faster learning; external verification).
AI response rule r = (1−α)·truth + α·(querier's own current belief) with
**stochastic rounding** so the delivered dose is linear in α
(manipulation check `effective_alpha`). Two configurations on identical
seeds: **main** = home-anchored mobility + spatially embedded bridged
communities + network-gated queries (friends + 2-hop); **control** =
immobile + disconnected communities + global query access (the structural
null). Design: 11 α-levels × 20 seed-paired replications × 200 ticks × 2
configurations; N=50 for α ∈ {0.8, 0.9, 1.0}; 300 ticks + policy switch
at tick 100 for the reversal experiment. Statistics: MixedLM (α linear +
quadratic × configuration, seed-grouped) with Holm-corrected per-level
paired contrasts. Metric guide (compendium §3): SECI per type + pop
(primary social); AECI-IE-chan per type + pop (primary AI-side);
AECI-LockIn + L1+ pool (individual lock-in); MAE on disaster cells,
unmet needs, precision (operational); retired variants SI-only. Full
ODD-style specification, calibration table, and metric formalism → SI.

## Limitations to state

Reduced-form α (a policy dial, not a trained system — no content, no
language, no personalisation beyond the revealed prior); stylized hazard
and relief process; N=100 agents (robust 100–500, but city-scale
extrapolation untested); exogenous, base-rate-limited verification
channel; a single uniform AI policy (no market of competing systems);
two discrete cognitive styles rather than a continuum (the gap sweep
spans the mix, not within-agent heterogeneity); no empirical calibration
to a specific disaster — the docking experiment anchors the dyadic core,
and transfer of the collective results is by extreme-case logic;
cross-configuration contrasts are structural comparisons, not evaluations
of any real platform's access policy; α\* transfers as "interior", not as
0.6. The E1 fit adds one: the model's confirmation-seekers adopt almost
none of an AI's disconfirming judgement at any certified parameter set,
so the model cannot reproduce the measured magnitude of AI-to-human
transmission for that type — it under-transmits AI influence, a
conservative mismatch stated as such.

## Provenance (dataset map)

Full map with runs, commits, and per-directory READMEs:
`RESULTS_COMPENDIUM.md` §1. The short version the writer needs:

- **`results-mechfix/`** — CANONICAL. Every steady-state number in the
  main text.
- `results-ablation-ownref/`, `results-ablation-detround/` — the
  attribution pair (H2).
- `results-salience/` — retrenchment counterfactual + C12 robustness.
- `results-reversal/` — hysteresis / repair / late-onset (H9).
- `results-lifecycle/` — per-seed lifecycle grid (H10; RNG-drift caveat).
- `results-boundary-n50/` — N=50 Holm-corrected boundary claims.
- `results-robustness/`, `results-gap-sweep-mechfix/`,
  `results-docking/` — envelope, 132-cell profile sweep, dyadic docking.
- `results/`, `results-verification/`, `results-final/`,
  `results-ablation-consensus/` — superseded history; SI transparency
  chain only.

Before submission: mint the Zenodo DOI over code + all results
directories; delete `docs/development/` from the public archive; verify
every quoted number against the compendium (and the compendium against
its source tables) one final time.

## Suggested skeleton

Written at the roomy length the tier-2/3 venues allow (Computers in
Human Behavior, J. R. Soc. Interface: no hard cap; for CHB, Methods
moves before Results); the same structure compresses to Nature Human
Behaviour Article length with Methods after Discussion. Target
~5,000–7,000 words main text, 5–6 main figures, no main-text tables
except the E4 predictions table in the Discussion (SI with a pointer
where the venue disallows it).

1. **Introduction** (~1,200–1,500 words — the NHB eight-paragraph arc,
   uncompressed): echo chambers → AI arrives, promise vs
   preference-trained sycophancy → disasters as the extreme case →
   beliefs form in networks → the dyadic-evidence gap and the research
   question → the model in one paragraph (with the type-agnostic
   design-principle sentence) → the four gaps → operationalization
   pointer (Fig. 1, concept).
2. **Results** (metrics lead-in of 2–3 sentences: sign convention,
   truth-convergence disambiguation, per-type default; one claim-first
   subsection per finding):
   2.1 Network-bounded access is the structural precondition (H3 + H7
       masking sentence; Fig. 2 configuration contrast).
   2.2 Confirming AI harms by starvation, and the optimum is interior
       (H4 + H1 + H2 attribution sentence + H7 U-shape; Fig. 3
       dose–response + optimum, starvation mechanism panel).
   2.3 The feedback loop: capture, lock-in, and the retrenchment
       counterfactual (H5 + H6; Fig. 4 capture/trust + salience cells).
   2.4 Harms concentrate on the spatial periphery and operations are
       buffered (H8; Fig. 5 periphery gaps + unmet needs).
   2.5 Dynamics: chambers form regardless, alignment blocks dissolution,
       and repair does not refill the pool (H9 + H10; Fig. 6 lifecycle +
       reversal trajectories) — promoted from the SI now that space
       allows; this is the governance-relevant finding.
3. **Discussion** (six paragraphs): mechanism synthesis
   (starves/captures; structure decides the failure mode; extends
   Glickman–Sharot to networks, Lorenz/Becker parallel); the descriptive
   optimum + non-confirmation adoption routes; retrenchment as boundary
   condition (widen the verified channel, don't just penalise the AI);
   the type-agnostic commitment and its ablation; monitoring implication
   (community-level convergence + belief-pool health, front-loaded harm);
   limitations + transfer logic.
4. **Methods** (~800–1,000 words, condensed from METHODS_PAPER.md per the
   facts section above).
5. **SI**: ODD specification; metric formalism incl. retired variants;
   calibration table; α\*-sensitivity/attribution table; N=50 boundary
   table; regression table; robustness envelope; docking figure;
   lifecycle per-seed grid; salience cells; transparency chain
   (legacy → fix → revision); promises/perils table.
