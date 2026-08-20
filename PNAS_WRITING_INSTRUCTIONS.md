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

## 2. Structure and key messages

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

**Introduction.** Preserve the four gaps from the NHB draft (structural
preconditions; alignment dose–response; belief–decision–feedback loop;
distributional consequences), each anchored in the existing citations. Add
one sentence stating the design principle now central to the model: *the AI
adapts only to what it can observe about a user — their revealed beliefs —
never to their cognitive type; all type differences emerge from human
information behavior.*

**Results — the four findings, updated key messages:**

1. *Network-bounded access is the structural precondition* (§F1 of
   RESULTS_OVERVIEW). Key message: under unrestricted access, high alignment
   *dissolves* the confirmation-seekers' community echo chamber (SECI ≈ +0.02);
   under network-gated access it persists (−0.27/−0.31); paired ΔSECI_exploit
   −0.29/−0.34 at α=0.9/1.0, CI excluding zero. Report SECI **per agent
   type** — the combined index conceals that the α-gradient is explorer-driven
   and the configuration contrast exploiter-driven.
2. *Confirming AI harms through starvation, not persuasion* (§F2). Key
   message: the accuracy cost of alignment falls entirely on accuracy-seekers
   (explorer MAE 0.55→1.75; exploiter MAE flat ≈1.8); mechanism: L1+ belief
   pool collapse 114→27 per agent. The interior optimum: α* ∈ [0.3, 0.6]
   across all composites in the main model; in the control, bubble-only
   composites select α*=1.0 — state the interiority claim with this
   qualifier.
3. *The feedback loop: capture and lock-in* (§F3). Key message: exploiters'
   AI query share rises 0.47→0.70 (capture); AI-heavy explorers' beliefs
   freeze (LockIn −0.04→−0.16); explorers do NOT discriminate against
   confirming AI at high α because verification is base-rate dominated
   (finding C12) — present this as a substantive finding about why
   accuracy-seeking fails, not a limitation. Network-bounded access buffers
   the operational collapse (unmet needs 3.1 vs 10.3 at α=1).
4. *Harms concentrate on the spatial periphery* (§F4). Key message: spatial
   MAE gap +0.13→+0.32 and aid gap −1.8→−6.6 with α in the main model; ≈0 in
   the immobile control (structural null).

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
metrics (SECI, AECI-Var/Err, LockIn, L1+ pool, MAE, unmet needs, precision),
experimental design (11 α × 20 seed-paired replications × 200 ticks × 2
configurations). Full ODD-style specification, calibration table, and metric
definitions go to the SI Appendix.

## 3. Display items

Main text (4 items):

- **Fig. 1 — Concept** (TikZ). Update from the NHB version: the AI box must
  show a single response rule for all callers (alignment toward the caller's
  revealed prior); the type-specific machinery (acceptance windows,
  confirmation vs accuracy rewards, verification) sits entirely on the human
  side. Show the two configurations as the structural contrast.
- **Fig. 2 — Configuration comparison**, regenerated from
  `results-verification/` (basis: `comparison_configs.png` + per-type SECI
  panels). Must display SECI **per agent type** — the exploiter dissolution
  vs persistence contrast is the finding.
- **Fig. 3 — Goldilocks composite** (panels a/b: alignment sweep composite
  with α* sensitivity spread; cognitive-gap sweep). Re-run the gap sweep on
  the fixed model before finalizing this figure.
- **Fig. 4 — Periphery** (spatial MAE and aid gaps vs α, main model vs
  control; basis `periphery_gap.png`).

SI Appendix (from existing S1–S9 plus new material; keep numbering stable
where possible): echo-chamber lifecycle; α* sensitivity tables (both
configurations); paired per-seed delta tables; **ablation section**
(consensus vs individual targeting: table of paired deltas + one paragraph);
**lock-in and L1+ pool figure** (the starvation mechanism); AI query share
and trust trajectories; legacy-vs-fixed transparency table
(RESULTS_OVERVIEW §7); robustness sweeps (Phase 2, once run); regression
tables (Phase 3); dyadic docking (Phase 3); promises/perils table (S10);
ODD description and calibration table.

Regeneration: dispatch **Compare Baseline vs Network/Mobility Switches**
(defaults now reproduce the fixed model) or **Replot Primary Sweep** from the
archived JSONs; never hand-edit figure data.

## 4. Style guidelines (carried over from the NHB draft, plus additions)

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

## 5. What to avoid (retired claims and known traps)

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
   bubble — it is structurally blind to own-prior confirmation. Use LockIn,
   query share, trust, and acceptance share for capture claims; use AECI-Err
   only with its heavy/light-split construct spelled out.
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

## 6. Submission checklist (author actions)

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
