# Writing Instructions — PNAS-family Manuscript (DisasterAIFilter)

*Revised 2026-08-25 after the mechanism revision (`30f89e0`) and its
ablation/counterfactual chain.*

Instructions for drafting the manuscript from the existing NHB draft
(`DisasterAIFilter_NHB.tex`; the ~3,000-word version at commit `2c2a09e` on
branch `claude/latex-merge-restructure-0x2rhf` is the preferred base).
**All results statements must be based on the revised-model canonical run**
(`results-mechfix/`, run 32821105202) — never on `results-final/`,
`results-verification/`, or legacy `results/` numbers (those appear only in
the SI transparency chain). Companion documents:
**`RESULTS_COMPENDIUM.md`** (every claim → dataset/figure/table) and
`RESULTS_OVERVIEW.md` §0.

## 1. Target venue — assessment and recommendation

| Priority | Journal | Route | Verdict |
|---|---|---|---|
| 1 | **PNAS Nexus** | Direct submission (PNAS format family) | **Recommended primary target** — see assessment below |
| 2 | PNAS | Direct Submission; Social Sciences → Psychological and Cognitive Sciences | Keep as stretch option, contingent on M6 docking + M2–M5 coming back clean |
| 3 | Journal of Computational Social Science / JASSS | Direct | Safety net; format rework is minor from the PNAS draft |

**Honest fit assessment (2026-08-25).** The scientific story is now
stronger than when PNAS Direct Submission was chosen: the interior optimum
is composite-robust (α\*=0.6, 7/12 definitions), attributable
(single-mechanism ablations), and accompanied by two genuinely
general-interest findings (the salience counterfactual's *social
retrenchment* — making disconfirmation salient ejects confirmation-seekers
into their network rather than toward truth — and the
fragmentation-vs-societal two-layer result). The Lorenz/Becker/Cinelli
lineage argument still holds. **Gate status (2026-08-26): the escalation condition is now FULLY
SATISFIED.** M6 docking PASS (`results-docking/` — Glickman & Sharot's
dyadic amplification reproduced for both types); M2 re-run DONE (interior
optimum in all 132 cognitive-profile cells; Fig. 3b reframed as
robustness); M3 boundary PASS (every high-α claim seed-robust at N=50
with Holm correction, `results-boundary-n50/`); M5 robustness PASS (all
three criteria at all ten perturbation levels, `results-robustness/` +
`docs/robustness_summary.md`); M7 decided; **M4 DONE**
(`results-mechfix/regression/` — Table S1/S6 material: significant
U-curvature on the operational outcomes, configuration main effects ***
throughout, Holm contrasts consistent with the N=50 boundary table).
**The full pre-drafting validation program is complete; drafting can
start.** The one remaining consideration against PNAS Direct Submission
is presentational: the late mechanism revisions mean the credibility case
rests on a substantial SI ablation chain, which competes with the 6-page
format. **Recommendation: with the full gate passed, PNAS Direct
Submission is now a defensible first target, with PNAS Nexus as the
transfer/fallback; the choice is the author's.** Draft to the PNAS format
either way. Drop NHB as a target: it
is harder than PNAS for this genre and the manuscript has moved away from
the NHB framing. The word budget, display-item limits, and style rules
below are PNAS-family rules and apply unchanged to Nexus.

Format constraints to write to (verify against the current PNAS author
guidelines at submission): main text ≈ **4,000–4,500 words** fitting the
6-page limit with display items; **4 main display items**; Abstract **≤250
words**; **Significance Statement ≤120 words** written for a general
scientific audience; Title ≤135 characters, no colons-with-clever-subtitle;
3–5 keywords; Materials & Methods as the final main-text section (concise,
with the full specification in the SI Appendix); single-PDF **SI Appendix**;
**American English** throughout (convert the NHB draft's British spellings:
behaviour→behavior, normalisation→normalization, etc.). Keep `sn-jnl`
compilable during drafting with a header comment noting the swap to
`pnas-new.cls` at submission. All citation keys must resolve against
`References.bib` in `tinacomes/DisasterAIFilterPaper`; add no new references.

## 2. Modelling steps — status after the 2026-08-25 mechanism revision

The list below preserves the original M1–M8 plan for traceability, with the
**current status** prepended per step. All remaining runs use the revised
model (`30f89e0`+: `confirmation_reference=network`,
`report_rounding=stochastic`, `confirmation_target=individual`) with the
standard seed-pairing (replicate *i* ← seed *i*).

- **M0′ (new, DONE)** — mechanism revision + attribution: exploiter
  network confirmation reference; stochastic report rounding (delivered
  dose linear in α); salience extended to the confirmation channel;
  population-level metric layer. Canonical dataset **`results-mechfix/`**;
  single-mechanism ablations `results-ablation-ownref/` (inert) and
  `results-ablation-detround/` (dose linearisation drives the robust
  interior α\*). Cite from `RESULTS_COMPENDIUM.md`.
- **M1 (RESOLVED, differently than planned)** — the belief-baseline
  AECI-IE did NOT validate as the primary AI-side index (≈0 for explorers
  in the main model pre-revision; exploiter series confounded; magnitude
  aggregation-fragile). The paper's AI-side pair is the **channel-baseline
  AECI-IE-chan (per type) + AECI-IE-chan-pop (population)**, with
  AECI-LockIn + L1+ pool as individual-lock-in evidence. The original M1
  spec below stands as design history only.
- **M2 (RE-RUN REQUIRED)** — the archived `results-gap-sweep-fixed/` is
  pre-revision; the step-function dose flattened the surface. Re-dispatch
  `run-gap-sweep.yml` on `30f89e0`+ before Fig. 3b exists.
- **M3, M4, M5, M6 (PENDING)** — unchanged in intent; run against
  `results-mechfix/` seeds and the revised model. M6 (dyadic docking) now
  also gates the venue decision (§1).
- **M7 (DECIDED, `results-salience/`)** — `salience_weight=0` stays
  mainline; C12 robust at every salience level; NEW finding: at s=1 the
  exploiter capture gradient vanishes via social retrenchment (chamber
  deepens −0.52 vs −0.37 at α=0; precision falls 0.43 vs 0.57). Goes to
  Finding 3 + Discussion, with **Table S8** (salience cells).
- **M8** — final figures from `results-mechfix/` (+ M2 re-run for
  Fig. 3b); Zenodo DOI over all results directories.

--- Original plan (design history; supersede numbers with the compendium) ---

**M1 — Metric revision: a SECI-comparable AI echo index (highest priority).**
Verdict on the current AECI family: **AECI-Var is inadequate** as the AI-side
counterpart of SECI. It groups agents by AI reliance and measures cross-agent
belief homogenization, so (a) it is structurally blind to the individualized
(own-prior) bubble the fixed model actually produces, and (b) it is
confounded at the truthful endpoint — empirically it is *most* negative at
α=0 (convergence on truth) and flattens toward α=1, the opposite of an
echo-chamber dose–response. AECI-Err is a harm metric (error split), not an
echo-structure metric; AECI-LockIn detects freezing but is activity- and
relief-confounded and not variance-comparable.

*Replacement — apply SECI's own construct to the served information.* SECI
compares belief variance inside a social exposure boundary (the community)
to global belief variance. The parallel AI-side construct applies the
**identical variance-ratio formula, asymmetric [−1,1] normalization, L1+
filter, per-community pooling, and per-type averaging** to the *report levels
community members receive from a channel* during the metrics window:

- `AECI-IE` (information-environment): pool of AI-delivered report levels
  received by community members; var(pool) vs var(global beliefs).
- `SECI-IE`: same over human-delivered report levels (consistency check —
  should track SECI per type).

Properties: at α=0 the AI serves the truth's diversity → AECI-IE ≈ 0 (no
truth-convergence confound); at α=1 it serves each caller's narrow priors →
strongly negative, capturing individualized confirmation and shared-source
convergence alike; the pair (SECI-IE, AECI-IE) is directly comparable and
`total_bubble = |SECI| + |AECI-IE|` becomes a sum of structurally identical
constructs (making the METHODS claim true).

Implementation: log `(channel, cell, level)` for every report received in
`seek_information`; aggregate every 5 ticks per community and type; report
pool sizes alongside (same shrinkage caveat as SECI); add both series to
`simulate.py` / `test_filter_bubbles.py` outputs and the α*-composite code
as a variant; extend `test_confirmation_target.py` with a smoke test
(AECI-IE ≈ −1 at α=1 for a converged caller; ≈ 0 at α=0). Then dispatch the
paired sweep (N=20, 200 ticks) → archive as `results-final/`. Metrics are
observation-only, so on identical seeds every existing series reproduces
exactly; the run only adds columns and **becomes the citable dataset**.
Validation criteria: AECI-IE ≈ 0 at α=0 and monotonically negative in α in
both configurations; SECI-IE tracks SECI per type; α* re-derived with the
new composite stays interior in the main model (report old- and new-composite
variants side by side in the α* sensitivity table).

**M2 — Cognitive-gap sweep on the fixed model** (feeds Fig. 3b). Dispatch
`run-gap-sweep.yml` (defaults now inherit `individual`); success: the
Goldilocks range remains a property of the population's cognitive profile;
archive `results-gap-sweep-fixed/`.

**M3 — Boundary strengthening.** Re-run α ∈ {0.8, 0.9, 1.0}, both
configurations, N=50 (extend `seed_base`); archive `results-boundary-n50/`.
Purpose: firm CIs for the Finding-1 paired deltas and the α ≥ 0.9 claims.

**M4 — Statistics upgrade.** `tools/sweep_regression.py`: per outcome (SECI
per type, AECI-IE, MAE per type, unmet needs, precision) fit α (linear +
quadratic) × configuration with seed as grouping factor (statsmodels MixedLM,
seed-clustered OLS fallback); standardized effects; Holm-corrected per-level
paired contrasts replacing raw CIs; output an SI-ready markdown table.

**M5 — Robustness envelope** (SI): four sweeps on the fixed model, N=20:
population {100, 300, 500} × α {0, 0.5, 0.7, 0.9, 1.0} (document community
scaling); one alternative within-community generator behind `--network_type`;
AI supply {1, 5, 10}; `verification_probability` {0.1, 0.3, 0.5} (report
recovery ticks alongside). One paragraph per sweep in
`docs/robustness_summary.md` stating whether the structural precondition,
interior optimum, and starvation/capture mechanisms hold.

**M6 — Dyadic docking** (SI): `experiments/dyadic_docking.py` — one human
(each type) × one AI, no network/relief, α sweep, vs a human–human pair;
success: qualitative reproduction of the dyadic amplification of Glickman &
Sharot (`glickman2024human`); one SI figure + paragraph.

**M7 — Salience decision experiment** (author decision): `salience_weight`
∈ {0, 0.5, 1} × α ∈ {0.7…1.0}; decides whether C12 stays a mainline finding
(recommended) or a salience>0 variant is promoted — Finding 3's text depends
on it.

**M8 — Final figures & packaging.** Regenerate Figs. 2–4 from
`results-final/`; build the SI figures (lock-in + L1+ pool; AI share/trust);
`.zenodo.json`; mint the DOI for code + all results directories.

## 3. Structure and key messages

Keep the **four-gaps → four-findings architecture** and the extreme-case
framing of disasters (AI-mediated information seeking under time pressure,
high stakes, and degraded verification). One Results subsection per finding.

**Significance (≤120 words).** Three sentences: (1) people increasingly ask
AI systems, not each other, what is happening — most consequentially in
disasters; (2) in a simulated disaster response, an AI that even mildly
confirms users' beliefs silently starves accuracy-seekers of information and
captures confirmation-seekers into self-reinforcing reliance, and whether
community echo chambers survive depends on the social network, not the AI
alone; (3) neither maximal truthfulness nor confirmation is operationally
optimal, which reframes AI alignment in crises as a socio-technical, not
purely technical, problem.

**Introduction (~900–1,100 words; compress the NHB intro's ~1,800).**
Paragraph-by-paragraph map — keep this order, merge as indicated, keep the
named citation keys (all resolve in `References.bib`):

- *¶1 — Echo chambers (background).* Societies fragment into segregated
  information spaces; definition of echo chambers and their association with
  polarization and erosion of shared factual ground
  (`mahmoudi2024echo`, `jeon2024hearhere`, `cinelli2021echo`,
  `bail2018exposure`, `baumann2020modeling`). Keep short — three sentences.
- *¶2 — AI arrives; promise vs construction (merge NHB ¶2+¶3).* Delegation of
  information collection and interpretation to AI (`klingbeil2024trust`,
  `angrisani2026gaps`); the bridging promise (`jeon2024hearhere`); then the
  collision: preference-based training on human approval
  (`christiano2017deep`, `ouyang2022training`) produces sycophancy
  (`sharma2023sycophancy`, `sharma2024generative`, `cheng2026sycophantic`),
  rejection of disconfirming information erodes trust (`glikson2020human`),
  and the alignment paradox (`west2025alignment`, `ekstrom2022self`). End
  with the operative question: how far should a system align to be used
  without forfeiting the correction it promises?
- *¶3 — Disasters as the extreme case (compress NHB ¶4).* Urgent decision
  situations (`svenson1993time`, `mendonca2001decision`,
  `comes2020coordination`); disasters concentrate time pressure, stakes, and
  degraded verification (`levin2012overcoming`, `sogaard2024evolution`);
  AI already deployed in crisis information work (keep two or three of
  `qadir2016crisis`, `reichstein2025early`, `acharya2025agentic`; move the
  rest to the SI promises/perils table S10). State the extreme-case logic
  explicitly: sharpest, most measurable form of the mechanism; misallocation
  translates directly into unmet needs.
- *¶4 — Beliefs form in networks (compress NHB ¶5).* Bounded confidence and
  homophily (`hegselmann2002opinion`, `mcpherson2001birds`), confirmation
  bias and selective exposure (`paulus2022influence`, `barbera2015tweeting`,
  `paulus2024interplay`), weak ties as the corrective channel
  (`granovetter1973strength`); crises tighten each mechanism — retreat to
  close groups, infrastructure disruption, silos, premature consensus
  (`comes2020coordination`, `holguin2012unique`, `pan2012crisis`,
  `driskell1991group`; sensemaking: `weick2005organizing`). Halve the NHB
  citation density here; keep one citation per mechanism.
- *¶5 — The evidence gap and the research question (NHB ¶6).* Dyadic
  human–AI feedback loops amplify individual bias (`glickman2024human`);
  collective accuracy is reshaped by network structure in ways dyadic
  analysis cannot reveal (`lorenz2011social`, `becker2017network`). Pose the
  question verbatim in spirit: does belief-aligned AI bridge or reinforce —
  and under which structural conditions?
- *¶6 — The model in one paragraph (NHB ¶7, updated).* ABM of decentralized
  disaster relief; two cognitive strategies (explore/exploit,
  `march1991exploration`); opinion-dynamics lineage (`deffuant2000mixing`,
  `hegselmann2002opinion`); Bayesian belief revision; learned source
  selection. **Add the design-principle sentence here:** *the AI adapts only
  to what it can observe about a user — their revealed beliefs, through the
  alignment parameter α — never to their cognitive type; all type
  differences emerge from human information behavior.*
- *¶7 — The four gaps, one compact paragraph each (compress NHB's four gap
  paragraphs by ~40%):* Gap 1 structural access conditions (bounded,
  spatially embedded access vs unrestricted digital access; the AI as an
  information source itself); Gap 2 the alignment dose–response (swift
  trust in crises: `tatham2010application`; adoption routes:
  `glikson2020human`, `lee2004trust`; whether harm is monotonic in α); Gap 3
  beliefs→decisions→feedback at population level (`gralla2016problem`);
  Gap 4 distribution of harms (remote decision-makers most dependent on
  mediated information: `holguin2012unique`; equity discourse centered on
  data bias, not network position: `coleman2024weaving`).
- *¶8 — Operationalization pointer.* One short paragraph mapping gaps to the
  design (two configurations on identical seeds; one finding per gap) and
  citing Fig. 1. In the Gap-2 sentence describing α, replace the NHB wording
  "confirmation of the beliefs held by the querier's community" with
  "confirmation of the querier's own prior beliefs".

**Results — four findings plus the societal layer: key messages and
exactly what goes where.** Each subsection cites its own display material
only; all numbers below are from **`results-mechfix/`** (verified against
`RESULTS_COMPENDIUM.md` §2 — re-check there before quoting).

1. *Network-bounded access is the structural precondition* (compendium H3).
   Key message: under unrestricted access, high alignment *dissolves* the
   confirmation-seekers' community echo chamber (SECI_exploit −0.45 →
   **+0.05**); under network-gated access it persists at every α (−0.39 …
   −0.18); the explorer chamber deepens 0 → −0.33 in both configurations.
   Combined paired ΔSECI at α=1: −0.140 [−0.211, −0.069] (per-type paired
   deltas from M4 replace this once available). Report SECI **per agent
   type** — the combined index conceals that the α-gradient is
   explorer-driven and the configuration contrast exploiter-driven.
   **Where:** cites **Fig. 2a** (exploiter SECI vs α, both configurations —
   the dissolution-vs-persistence crossing IS this figure's job) and
   **Fig. 2b** (explorer SECI, both configurations, showing the shared
   deepening); paired deltas with Holm-corrected CIs in **Table S1**
   (from M4); lifecycle/evolution in **Fig. S1** (basis:
   `population_evolution.png`, per-type columns). Inline numbers: the two
   SECI_exploit endpoints, the paired delta with CI.
2. *Confirming AI harms through starvation, not persuasion* (compendium
   H4 + H1). Key message: the accuracy cost of alignment falls entirely on
   accuracy-seekers (explorer MAE 0.54→1.74; exploiter MAE flat ≈1.8), and
   the dose–response is smooth in α (delivered dose linear — cite
   `effective_alpha` once); mechanism: L1+ belief pool collapse 113→29 per
   agent. Interior optimum: **α\* = 0.6 for 7/12 composite definitions**
   in the main model (spread [0.1, 0.6]); control bubble composites
   0.8–0.9 — interior in BOTH configurations (the old control corner
   solution was a dose artifact; state this in one sentence with the
   ablation citation).
   **Where:** cites **Fig. 2c** (per-type MAE vs α, both configurations)
   and **Fig. 3a** (composite vs α with the α\* marks; population
   composite as the headline curve, per-type composite overlaid); the
   starvation mechanism cites **Fig. S2** (L1+ pool + LockIn vs α); α\*
   sensitivity across all 12 variants (incl. retired AECI-Var and the two
   single-mechanism ablations) in **Table S2**. Inline numbers: MAE
   endpoints per type, pool collapse 113→29, α\*=0.6 with the 7/12 count.
3. *The feedback loop: capture, lock-in — and the retrenchment
   counterfactual* (compendium H5 + H6). Key message: exploiters' AI query
   share rises 0.54→0.69 (capture) while their AI trust stays flat ≈0.47;
   AI-heavy explorers' beliefs freeze (LockIn −0.01→−0.13); explorers do
   NOT discriminate against confirming AI at high α because verification
   is base-rate dominated (C12: trust 0.88→0.84, and 0.90→0.82 even under
   full salience — robust). NEW counterfactual result: making
   disconfirmation salient (s=1) eliminates the capture gradient (AI share
   flat ≈0.5) but produces **social retrenchment**, not accuracy — at the
   truthful endpoint the exploiter chamber deepens (SECI −0.52 vs −0.37)
   and precision falls (0.43 vs 0.57). Network-bounded access buffers the
   operational collapse (unmet needs 2.86 vs 10.18 at α=1; explorer
   precision 0.62 vs 0.20).
   **Where:** cites **Fig. 2d** (unmet needs vs α, both configurations),
   **Fig. S3** (AI query share and trust trajectories per type; the C12
   flat explorer-trust curve is panel S3b), **Table S8** (salience cells,
   from `results-salience/`); the optimum's robustness to the
   population's cognitive profile cites **Fig. 3b** (M2 re-run: interior
   α\* in every (g, d_mid) cell — a robustness claim, NOT a dependence
   claim; see the Fig. 3b spec in §4). Inline numbers: query-share
   endpoints, unmet-needs contrast, the two salience contrasts.
4. *Harms concentrate on the spatial periphery* (compendium H8). Key
   message: spatial MAE gap +0.12→+0.33 and aid gap −1.8→−6.6 with α in
   the main model; ≈0 at all α in the immobile control (structural null);
   betweenness/broker gaps small — the periphery is spatial, not
   graph-positional, under mobility.
   **Where:** cites **Fig. 4a** (spatial MAE gap vs α, both
   configurations) and **Fig. 4b** (aid-contribution gap vs α); within-run
   gap evolution in **Fig. S4**; network-centrality decomposition in
   **Table S3**. Inline numbers: the two gap endpoints per configuration.
5. *The societal layer* (compendium H7) — woven into Findings 1–2, not a
   separate subsection: (a) one sentence in Finding 1 that the
   population-level SECI **masks** the per-type crossing (−0.22 → 0.00 in
   the control) — the fragmentation lens is not optional; (b) one sentence
   in Finding 2 that the population information-environment index
   (AECI-IE-chan-pop) is U-shaped in α in both configurations (shallowest
   ≈ α=0.7), aligning the served-information evidence with the operational
   optimum; the population composites are also the most ablation-robust
   α\* definition (0.6 in all three revised-model datasets). **Where:**
   both sentences cite **Fig. S5** (basis: `population_evolution.png`,
   population column).

**Discussion.** (a) The mechanism synthesis: sycophantic AI *captures*
confirmation-seekers and *starves* accuracy-seekers; social structure decides
whether community chambers survive and whether operations collapse. (b) The
interior optimum is a **descriptive property of the trade-off, not a design
recommendation** — name adoption routes that need no confirmation
(uncertainty communication, transparency, verification support). (c) The
type-agnostic AI as a modeling commitment and its ablation (one paragraph:
the confirmation target is behaviorally inert; findings do not depend on the
AI knowing anything unobservable). (d) Limitations: reduced-form α, stylized
hazard, N=100, exogenous verification channel, single uniform AI policy;
cognitive-gap sweep caveat for transferring α* across populations.

**Materials & Methods (main text, concise).** Agent types (D/δ, learning
rates, evaluation channels — including that exploiters score confirmation
against their **trusted-network consensus** when defined, own prior
otherwise: `confirmation_reference=network`), the alignment formula
r = (1−α)t + αb with **b = the querier's own current belief for both
types** and **stochastic rounding of the delivered report** (dose linear
in α; manipulation check `effective_alpha`), network/mobility switches,
metrics (SECI per type + population; AECI-IE-chan per type + population
with the SECI-IE-chan consistency check; LockIn; L1+ pool; MAE; unmet
needs; precision; AECI-Err/AECI-Var as retired variants in the SI),
experimental design (11 α × 20 seed-paired replications × 200 ticks × 2
configurations). Full ODD-style specification, calibration table, and
metric definitions go to the SI Appendix.

## 4. Display items — panel-level specifications

Main text (4 items, all figures; no main-text tables — every table is SI):

- **Fig. 1 — Concept** (TikZ, reworked from the NHB version). Required
  changes: the AI box states the single response rule for ALL callers —
  `r = (1−α)t + αb`, "α=1: confirms the **querier's own prior**" (the NHB
  TikZ says "confirms the community's beliefs" — must change, node and
  caption); the human box carries the type-specific machinery (acceptance
  windows D/δ, confirmation vs accuracy rewards, external verification);
  outcomes box lists SECI, AECI-IE (replacing AECI), MAE, unmet needs,
  precision. Keep the Gap badges and the two-configuration contrast.
- **Fig. 2 — Configuration comparison** (from **`results-mechfix/`**; basis
  `comparison_configs.png` rebuilt with per-type panels). Four panels:
  **(a)** SECI_exploit vs α, main vs control — the dissolution/persistence
  contrast; **(b)** SECI_explor vs α, main vs control — the shared
  deepening; **(c)** MAE per type vs α, both configurations (exploiter
  flatness visible); **(d)** unmet needs vs α, both configurations. Shared
  α-axis; seed-robust α-levels marked; sign convention in the caption.
- **Fig. 3 — Goldilocks** two panels: **(a)** composite vs α (main model)
  computed with the **population composite |SECI-pop| + |AECI-IE-chan-pop|**
  as the headline curve (the most ablation-robust definition, α\*=0.6 in
  all three revised-model datasets), the per-type channel-baseline
  composite overlaid, and the control's curve for contrast (both now
  interior — the caption notes the old corner solution was a dose
  artifact, citing the rounding ablation); **(b)** cognitive-gap sweep
  (from `results-gap-sweep-mechfix/`, all 132 cells) — **REFRAMED as a
  robustness panel**: α\* (population bubble and unmet-needs) across the
  (g, d_mid) grid, showing the interior optimum is invariant to the
  population's cognitive profile (0.6–0.8 in every cell). Do NOT claim
  the optimum's location depends on the cognitive profile — the M2 re-run
  does not support it at the population/operational level; the per-type
  drift with g (0.2–0.5 → 0.6–0.7) may be mentioned as suggestive only
  with M4 CIs.
- **Fig. 4 — Periphery** two panels: **(a)** spatial MAE gap (far − near
  quartile) vs α, main vs control; **(b)** aid-contribution gap vs α, same
  overlay; basis `periphery_gap.png` from `results-mechfix/`.

SI Appendix numbering (fixed here so Results can cite it):

- **Fig. S1** echo-chamber lifecycle/evolution (basis:
  `population_evolution.png` per-type columns + `echo_chamber_lifecycle.png`).
- **Fig. S2** starvation mechanism: L1+ belief pool per type and AECI-LockIn
  vs α, both configurations.
- **Fig. S3** feedback loop: (a) AI query share per type vs α; (b) AI trust
  per type vs α (the C12 flat explorer curve).
- **Fig. S4** periphery gap evolution within runs.
- **Fig. S5** population vs per-type bubble evolution (the societal layer;
  basis: `population_evolution.png`, all three rows).
- **Fig. S6** dyadic docking (M6). **Fig. S7** robustness sweeps (M5).
- **Fig. S8** alignment-reversal experiment (M9; basis:
  `results-reversal/reversal-tables/reversal_trajectories.png`) — the
  repair/late-onset trajectories against constant anchors.
- **Table S1** paired per-seed deltas (main − control) with Holm-corrected
  CIs, all outcomes (M4).
- **Table S2** α* sensitivity across all 12 composite variants, both
  configurations, PLUS the two single-mechanism ablation columns
  (own-reference; deterministic rounding) — the attribution table.
- **Table S3** network-centrality/broker decomposition.
- **Table S4** ablation: consensus − individual paired deltas + verdict
  paragraph (AI-side targeting; unchanged).
- **Table S5** transparency chain: legacy → type-agnostic fix → mechanism
  revision, one row per headline quantity (extend RESULTS_OVERVIEW §7 with
  a `results-mechfix` column).
- **Table S6** regression table (M4). **Table S7** calibration table
  (parameter → empirical source).
- **Table S8** salience experiment cells (s × α; from
  `results-salience/salience-tables/robustness_tables.md`) — supports the
  C12-robustness and social-retrenchment sentences.
- **Table S10** promises/perils (kept).
- **Table S11** per-seed lifecycle + reversal statistics (formation,
  peak, dissolution-by-horizon, capture onset at the M9 anchors;
  post-switch recovery and endpoint contrasts; from
  `results-reversal/`). Supports the Finding-2 lifecycle claim, the
  Finding-3 capture-onset and reversal sentences, and the Discussion's
  monitoring paragraph. 2026-08-27 dynamics revision: the paper now
  carries the lifecycle claims ("alignment blocks dissolution, not
  depth"; capture-onset acceleration; reversal hysteresis + late-onset
  asymmetry; periphery non-equilibration) — all headline dynamic
  numbers cite `results-reversal/` (N=20, CI-grade) or SI Fig. S1/S4
  (mean-trajectory); the full per-seed lifecycle grid lands with the
  instrumented primary-sweep re-run (lc_*_runs columns).
- ODD-protocol model description as the SI's Methods complement.

Regeneration: dispatch **Compare Baseline vs Network/Mobility Switches**
(defaults reproduce the revised model) or **Replot Primary Sweep** from the
archived JSONs; never hand-edit figure data.

## 5. Style guidelines (carried over from the NHB draft, plus additions)

- **No figuresplaining.** "Confirming AI deepens the explorer communities'
  convergence (Fig. 2c)" — never "Fig. 2c shows that…".
- Results in present tense; one claim per paragraph; the paragraph's first
  sentence *is* the claim.
- Numbers: report steady-state means with across-replication SE or 95% CI;
  call an effect seed-robust **only** when the paired per-seed CI excludes
  zero; two significant decimals for indices, whole numbers ≥10.
- Sign conventions stated once and used consistently (negative SECI/AECI =
  echo chamber); indices always named with their construct on first use.
- Per-type reporting is the default for SECI, MAE, trust, and query shares;
  combined indices only where the composite requires them, and never to
  conceal a per-type divergence.
- Mechanism sentences must name the *agent-side* process (acceptance,
  verification, reward, starvation, capture) — the AI has one behavior;
  everything differential is human.
- Keep the extreme-case framing paragraph (disasters as the stress test for
  AI-mediated information seeking); keep the four-gap scaffolding verbatim in
  spirit.
- Significance and Abstract must be readable by a non-modeler; no acronyms in
  the Significance Statement (spell out "echo chamber", avoid SECI/AECI/MAE).

## 6. What to avoid (retired claims and known traps)

1. **Do not** attribute the SECI deepening to "consensus recycling" /
   "the AI targets the consensus belief of the caller's trusted network for
   exploitative queriers" (old NHB §Results line ~238 and Methods ~341).
   That mechanism text described legacy code, was contradicted by the
   per-type data, and the ablation shows the targeting is inert. Delete or
   rewrite wherever it appears.
2. **Do not** describe the AI as treating agent types differently anywhere —
   including figure captions and the SI. The model's design principle is the
   opposite, and a smoke test enforces it (`test_confirmation_target.py`).
3. **Do not** use AECI-Var as evidence for (or against) an individualized AI
   bubble — it is structurally blind to own-prior confirmation and is most
   negative at the truthful endpoint (truth-convergence confound). The
   AI-side echo construct in the paper is **AECI-IE-chan (channel
   baseline), per type and population** — NOT the belief-baseline AECI-IE,
   which failed its own validation for exploiters and is aggregation-fragile
   (see M1 status in §2); belief-baseline AECI-IE may be cited only for the
   explorer dose–response in the control. AECI-Var appears only in the α*
   sensitivity table as a retired variant. Capture claims use LockIn, query
   share, trust, and acceptance share; AECI-Err only with its
   heavy/light-split construct spelled out.
4. **Do not** present SECI deepening without the L1+ pool context: part of
   the explorer deepening is pool shrinkage (belief starvation), not
   convergence on a shared false narrative. Say so explicitly once.
5. **Do not** quote numbers from `results/`, `results-verification/`, or
   `results-final/` in the main text — all main-text numbers come from
   `results-mechfix/` (check against `RESULTS_COMPENDIUM.md`); the older
   runs appear only in the SI transparency chain (Table S5).
6. **Do not** report a single naked α*: report **α\*=0.6 with its 7/12
   composite count and the [0.1, 0.6] sensitivity spread**, plus the
   ablation attribution (Table S2); never compare range-normalized
   composites across configurations (use raw paired deltas).
6b. **Do not** describe the old control-configuration corner solution
   (α\*=1.0, "full confirmation minimises the bubble indices") as a model
   property — the ablations show it was a dose-delivery artifact of
   deterministic rounding; both configurations are interior under the
   linear dose.
6c. **Do not** present the salience counterfactual as a remedy: at s=1 the
   capture gradient disappears via social retrenchment (deeper chamber,
   worse precision at the truthful endpoint), not via truth-seeking. Frame
   as a boundary condition of the capture mechanism.
7. **Do not** frame the interior optimum as a recommendation to build mildly
   confirming AI (Discussion must carry the explicit disclaimer and the
   non-confirmation adoption routes).
8. **Do not** claim explorers "learn to distrust" confirming AI at high α —
   they don't (C12); the failure of accuracy-seeking under base-rate-dominated
   verification is itself a finding.
9. **Do not** add references beyond `References.bib`; do not resurrect the
   removed "Fu et al. 2023" claim without resolving its TODO; resolve the
   `steinbrink2024transparency` placeholder before submission.
10. Avoid: overclaiming external validity (reduced-form α ≠ any deployed
    system); "prove/optimal policy" language; unqualified "echo chamber"
    for the truthful-AI convergence at α=0 (the NHB draft's accuracy-metric
    disambiguation paragraph must survive the compression).

## 7. Submission checklist (author actions)

1. Decide the venue per §1 (recommended: PNAS Nexus; escalate to PNAS only
   if M6 docking reproduces Glickman & Sharot and M2–M5 hold).
2. Approve Significance wording; complete Acknowledgements and Funding.
3. Mint the Zenodo DOI (code + ALL results directories:
   `results-mechfix/`, `results-ablation-ownref/`,
   `results-ablation-detround/`, `results-salience/`, plus the historical
   chain `results/`, `results-verification/`,
   `results-ablation-consensus/`, `results-final/`); add to Data/Code
   availability.
4. Cover letter: lead with "extends the dyadic human–AI feedback-loop
   findings (Glickman & Sharot, NHB 2024) to networked populations; closest
   methodological ancestors Lorenz 2011 / Becker 2017 (both PNAS)"; suggest
   3–5 reviewers from computational social science / collective intelligence
   / crisis informatics.
5. Verify current PNAS/Nexus limits (title/abstract/significance/pages) and
   swap in the journal class at submission.
