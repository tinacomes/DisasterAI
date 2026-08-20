# Writing Instructions — PNAS Manuscript (DisasterAIFilter)

Instructions for drafting `DisasterAIFilter_PNAS.tex` from the existing NHB
manuscript (`DisasterAIFilter_NHB.tex`; the ~3,000-word version at commit
`2c2a09e` on branch `claude/latex-merge-restructure-0x2rhf` is the preferred
base). **All results statements must be based on the fixed-model verification
run** (`results-verification/`, run 32298278561) — never on the legacy numbers
in `results/`. Companion document: `RESULTS_OVERVIEW.md` (citable numbers).

## 1. Target journals

| Priority | Journal | Route | Rationale |
|---|---|---|---|
| 1 | **PNAS** | Direct Submission; Social Sciences → Psychological and Cognitive Sciences; optional dual classification Physical Sciences → Computer Sciences | Theory-building simulation without new empirical data is a PNAS genre; the paper's closest methodological ancestors (Lorenz et al. 2011; Becker et al. 2017) and its echo-chamber measurement reference (Cinelli et al. 2021) are PNAS papers |
| 2 | PNAS Nexus | Direct submission | Same audience and format family; faster; accepts transfers |
| 3 | Nature Human Behaviour | Full resubmission (previous target) | Fits the Glickman & Sharot (2024) lineage; requires reverting to NHB format |

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

## 2. Remaining modelling steps (M1–M8) — complete before drafting Results

Ordered plan; M1 defines the final citable dataset and gates everything else.
M2/M3 can run in parallel after M1; M4 consumes M1+M3; M5–M7 are SI material;
M8 is last. All runs use the fixed model (`55d4b2b`+, `confirmation_target=
individual` default) with the standard seed-pairing (replicate *i* ← seed *i*).

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

**Results — the four findings: key messages and exactly what goes where.**
Each subsection cites its own display material only; numbers below are from
`results-verification/` and must be re-extracted from `results-final/` after
M1 (expected identical for existing metrics; AECI-IE values are new).

1. *Network-bounded access is the structural precondition* (§F1 of
   RESULTS_OVERVIEW). Key message: under unrestricted access, high alignment
   *dissolves* the confirmation-seekers' community echo chamber (SECI ≈ +0.02);
   under network-gated access it persists (−0.27/−0.31); paired ΔSECI_exploit
   −0.29/−0.34 at α=0.9/1.0, CI excluding zero. Report SECI **per agent
   type** — the combined index conceals that the α-gradient is explorer-driven
   and the configuration contrast exploiter-driven.
   **Where:** cites **Fig. 2a** (exploiter SECI vs α, both configurations —
   the dissolution-vs-persistence crossing IS this figure's job) and
   **Fig. 2b** (explorer SECI, both configurations, showing the shared
   deepening); paired deltas with Holm-corrected CIs in **Table S1**
   (from M4); lifecycle in **Fig. S1**. Inline numbers: the two SECI_exploit
   endpoints, the two paired deltas with CIs.
2. *Confirming AI harms through starvation, not persuasion* (§F2). Key
   message: the accuracy cost of alignment falls entirely on accuracy-seekers
   (explorer MAE 0.55→1.75; exploiter MAE flat ≈1.8); mechanism: L1+ belief
   pool collapse 114→27 per agent; the AI-channel echo index (AECI-IE, M1)
   deepens with α while ≈0 at the truthful endpoint. Interior optimum:
   α* ∈ [0.3, 0.6] across composites in the main model; in the control,
   bubble-only composites select α*=1.0 — state interiority with this
   qualifier.
   **Where:** cites **Fig. 2c** (per-type MAE vs α, both configurations) and
   **Fig. 3a** (composite vs α with the α* spread band; control curve
   overlaid); the starvation mechanism cites **Fig. S2** (L1+ pool + LockIn
   vs α); α* sensitivity across composite variants (including the retired
   AECI-Var variant for transparency) in **Table S2**. Inline numbers: MAE
   endpoints per type, pool collapse 114→27, α* spread.
3. *The feedback loop: capture and lock-in* (§F3). Key message: exploiters'
   AI query share rises 0.47→0.70 (capture); AI-heavy explorers' beliefs
   freeze (LockIn −0.04→−0.16); explorers do NOT discriminate against
   confirming AI at high α because verification is base-rate dominated
   (finding C12) — a substantive finding about why accuracy-seeking fails,
   not a limitation. Network-bounded access buffers the operational collapse
   (unmet needs 3.1 vs 10.3 at α=1; exploiter precision 0.37 vs 0.22).
   **Where:** cites **Fig. 2d** (unmet needs vs α, both configurations) and
   **Fig. S3** (AI query share and AI trust trajectories per type; the C12
   flat explorer-trust curve is panel S3b); cognitive-gap dependence of the
   optimum cites **Fig. 3b** (from M2). Inline numbers: query-share
   endpoints, unmet-needs contrast, explorer trust at α=1.
4. *Harms concentrate on the spatial periphery* (§F4). Key message: spatial
   MAE gap +0.13→+0.32 and aid gap −1.8→−6.6 with α in the main model; ≈0 in
   the immobile control (structural null); betweenness/broker gaps small —
   the periphery is spatial, not graph-positional, under mobility.
   **Where:** cites **Fig. 4a** (spatial MAE gap vs α, both configurations)
   and **Fig. 4b** (aid-contribution gap vs α); within-run gap evolution in
   **Fig. S4**; network-centrality decomposition in **Table S3**. Inline
   numbers: the two gap endpoints per configuration.

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
rates, evaluation channels), the alignment formula r = (1−α)t + αb with **b =
the querier's own current belief for both types**, network/mobility switches,
metrics (SECI, AECI-IE with its SECI-IE consistency check, LockIn, L1+ pool, AECI-Err as secondary, MAE, unmet needs, precision),
experimental design (11 α × 20 seed-paired replications × 200 ticks × 2
configurations). Full ODD-style specification, calibration table, and metric
definitions go to the SI Appendix.

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
- **Fig. 2 — Configuration comparison** (from `results-final/`; basis
  `comparison_configs.png` rebuilt with per-type panels). Four panels:
  **(a)** SECI_exploit vs α, main vs control — the dissolution/persistence
  contrast; **(b)** SECI_explor vs α, main vs control — the shared
  deepening; **(c)** MAE per type vs α, both configurations (exploiter
  flatness visible); **(d)** unmet needs vs α, both configurations. Shared
  α-axis; seed-robust α-levels marked; sign convention in the caption.
- **Fig. 3 — Goldilocks** two panels: **(a)** total_bubble and total_score
  vs α (main model) with the α* spread band [0.3, 0.6] and the control's
  bubble-composite curve overlaid (α*=1.0 divergence visible), computed with
  the M1 composite (|SECI| + |AECI-IE|); **(b)** cognitive-gap sweep (from
  M2): Goldilocks-range location as a function of the population's cognitive
  profile.
- **Fig. 4 — Periphery** two panels: **(a)** spatial MAE gap (far − near
  quartile) vs α, main vs control; **(b)** aid-contribution gap vs α, same
  overlay; basis `periphery_gap.png` from `results-final/`.

SI Appendix numbering (fixed here so Results can cite it; existing S1–S9
content keeps its identity where possible):

- **Fig. S1** echo-chamber lifecycle (formation/persistence/recovery).
- **Fig. S2** starvation mechanism: L1+ belief pool per type and AECI-LockIn
  vs α, both configurations.
- **Fig. S3** feedback loop: (a) AI query share per type vs α; (b) AI trust
  per type vs α (the C12 flat explorer curve).
- **Fig. S4** periphery gap evolution within runs.
- **Fig. S5** dyadic docking (M6). **Fig. S6+** robustness sweeps (M5).
- **Table S1** paired per-seed deltas (main − control) with Holm-corrected
  CIs, all outcomes (M4).
- **Table S2** α* sensitivity across composite variants, both configurations,
  including the retired AECI-Var variant for transparency.
- **Table S3** network-centrality/broker decomposition.
- **Table S4** ablation: consensus − individual paired deltas + verdict
  paragraph.
- **Table S5** legacy-vs-fixed transparency table (RESULTS_OVERVIEW §7).
- **Table S6** regression table (M4). **Table S7** calibration table
  (parameter → empirical source). **Table S10** promises/perils (kept).
- ODD-protocol model description as the SI's Methods complement.

Regeneration: dispatch **Compare Baseline vs Network/Mobility Switches**
(defaults now reproduce the fixed model) or **Replot Primary Sweep** from the
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
   AI-side echo construct in the paper is **AECI-IE** (M1); AECI-Var appears
   only in the α* sensitivity table as a retired variant. Capture claims use
   LockIn, query share, trust, and acceptance share; AECI-Err only with its
   heavy/light-split construct spelled out.
4. **Do not** present SECI deepening without the L1+ pool context: part of
   the explorer deepening is pool shrinkage (belief starvation), not
   convergence on a shared false narrative. Say so explicitly once.
5. **Do not** quote legacy-run numbers in the main text (they differ in AI
   usage levels); the legacy run appears only in the SI transparency table.
6. **Do not** report a single α*: report the composite spread ([0.3, 0.6]
   main model) and the sensitivity table; never compare range-normalized
   composites across configurations (use raw paired deltas).
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

1. Approve Significance wording; complete Acknowledgements and Funding.
2. Mint the Zenodo DOI (code + `results/`, `results-verification/`,
   `results-ablation-consensus/`); add to Data/Code availability.
3. Cover letter: lead with "extends the dyadic human–AI feedback-loop
   findings (Glickman & Sharot, NHB 2024) to networked populations; closest
   methodological ancestors Lorenz 2011 / Becker 2017 (both PNAS)"; suggest
   3–5 reviewers from computational social science / collective intelligence
   / crisis informatics.
4. Verify current PNAS limits (title/abstract/significance/pages) and swap in
   `pnas-new.cls`.
