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

## Target venue (three options; author decision)

The paper is theory- and model-based with no empirical data of its own;
its external-validity anchor is the docking reproduction of a published
dyadic experiment (H-docking, compendium §4 M6). The emphasis is
AI–human interaction at population scale, and the author's goals are
**readership and discussion**. Excluded: PNAS/PNAS Nexus (author
decision); JASSS and J. Computational Social Science
(social-simulation audience, not the AI + behaviour audience);
Collective Intelligence and other young non-indexed venues (author
decision after a prior submission experience). All three options below
are established, indexed, high-visibility journals.

| Priority | Journal | Fit | Main risk |
|---|---|---|---|
| 1 | **Science Advances** | Direct precedent for pure-simulation studies of AI social dynamics (e.g. *Emergent social conventions and collective bias in LLM populations*, 2025) and for opinion-dynamics ABMs; the Lorenz/Becker/Glickman lineage argument carries; broad interdisciplinary readership; roomy format (no 6-page squeeze, generous SI) | High bar; the cover letter must make the machine-behaviour framing and the docking anchor do real work |
| 2 | **Nature Communications** | Publishes computational social science ABMs on algorithmic/AI amplification; enormous reach and indexing; the "human–machine social systems" framing is at home in the Nature portfolio | High APC; slower reviews; its computational-social-science bar for simulation-only papers is high |
| 3 | **Nature Machine Intelligence** | The journal route to the AI community proper; sycophancy/alignment framing is timely there; Nature-portfolio transfer (e.g. to Nat Comms) softens rejection cost | Leans ML-methods for primary research; an ABM of *human* behaviour is a scope stretch — send a presubmission inquiry first |

**Recommendation: Science Advances**, framed as machine behaviour /
collective human–AI epistemics with the dyadic docking as the empirical
anchor. **And independent of venue: post the preprint to arXiv (cs.CY +
cs.MA, cross-list physics.soc-ph) at submission** — the AI community
discovers work through arXiv, not journal tables of contents; this is
how the paper reaches that audience whichever journal carries it.

**Contingency — if E2 is implemented** (the α-measurement of deployed
assistants; see *Empirical foundations* below): the paper gains primary
empirical content on real AI systems. The recommendation stays Science
Advances, but NMI upgrades from scope-stretch to a credible co-equal
first choice for AI-community reach — measured sycophancy of deployed
systems plus its population-scale consequences is squarely NMI's genre.
Resolve by presubmission inquiry to NMI while preparing the Science
Advances package.

**What the venue change buys the redraft:** at Science Advances the PNAS
corset is gone — no 6-page limit, no 4-display-item cap, no Significance
Statement. Write ~5,000–7,000 words of main text with 5–6 main figures,
and pull the credibility chain (dose-linearisation ablation, docking,
salience counterfactual) *into the main text* instead of burying it in
the SI — that compression was the standing weakness of the PNAS draft.
(Nature Comms/NMI would re-impose a ~5,000/3,500-word ceiling with
Methods after Discussion — the skeleton below compresses back if the
author picks those.) Keep American English.

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
| — | Docking (external-validity anchor) | Glickman–Sharot dyadic amplification reproduced for both types; aligned AI retains more bias than a human partner | `results-docking/` (M6) |
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
   Sharot's human–AI bias amplification for both cognitive types. The
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
reproduction, *not* parameter-matched calibration (SI §M6); (ii) two
empirically sourced entries in the calibration table (returner–explorer
mobility, Pappalardo et al.; weak-tie structure, Granovetter); (iii)
design-rationale justifications for everything else. The steps below are
ordered by cost. E1–E4 are pre-submission work for THIS paper; E5–E6
strengthen it if time allows; E7–E8 are the follow-up programme and enter
the paper only as the stated research agenda in the Discussion — the
paper is not held for them.

**Tier 1 — pre-submission (weeks, no new human data):**

- **E1 — Parameter-matched docking** (highest leverage). Upgrade M6 from
  qualitative to fitted: use Glickman & Sharot's published effect sizes
  to estimate the acceptance/trust parameters (D, δ, learning rates) in
  the dyad (`experiments/dyadic_docking.py` is the harness; wrap it in a
  fitting loop), then show the population runs under fitted vs. default
  parameters. Buys: "micro-parameters empirically estimated," turning
  the docking section from consistency check into indirect calibration.
- **E2 — Locate deployed systems on the α scale.** Measure the
  confirmation weight of current AI assistants with the model's own
  estimand (the mixing weight in r = (1−α)t + αb) under a severity-
  estimation protocol, and place them as bands on the paper's α axis.
  **Full protocol: `docs/ALPHA_MEASUREMENT_PROTOCOL.md`.** Buys: the
  paper's single biggest empirical upgrade — α stops being a free
  abstraction; the dose–response curves acquire a "you are here" marker.
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
  show the mechanisms are not artifacts of hand-coded rules; direct
  Science Advances precedent for the genre.

**Venue coupling:** implementing E2 changes the paper's genre from
simulation-only to simulation + measurement of deployed systems — see
the contingency note in the venue section.

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
0.6.

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

Written for the recommended venue (Science Advances; the same structure
compresses to Nature Communications / NMI length, which also move Methods
after Discussion). Target ~5,000–7,000 words main text, 5–6 main
figures, no main-text tables.

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
