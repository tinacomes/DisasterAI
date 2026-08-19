# Submission plan: PNAS (recommended route)

Target: PNAS Direct Submission, Social Sciences (Psychological and Cognitive
Sciences; dual classification Physical Sciences / Computer Sciences optional).

**Why PNAS**: the paper is a theory-building simulation without an empirical
dataset. PNAS publishes exactly this genre (Lorenz et al. 2011 and Becker et
al. 2017 — the paper's own key references — are PNAS papers). No new data
collection is required; the gaps are closed with ablations, robustness runs,
statistics, literature-based calibration, and a dyadic docking exercise.

## Repo map (context for every Claude Code session)

- Model repo: `tinacomes/DisasterAI` (this repo). Core: `DisasterAI_Model.py`
  (~4,000 lines; `HumanAgent`, `AIAgent` at line ~2068; alignment applied in
  the AI response path ~lines 2117–2240; trusted-network consensus targeting
  for exploitative callers uses `HumanAgent.get_network_consensus`, line ~658).
- CLI: `simulate.py` with `--alpha --n_runs --seed_base --mobility {0,1}
  --network_type {components,...} --query_scope {global,network}
  --share_exploitative --disaster_dynamics --rumor_probability
  --salience_weight`.
- GitHub Actions: `.github/workflows/run-primary-sweep.yml`,
  `compare-network-mobility.yml`, `run-gap-sweep.yml`, `run-factor-sweeps.yml`,
  `replot-primary.yml`, `replot-gap-sweep.yml`.
- Archived results: `results/` (provenance in `results/README.md`; current
  numbers from run 32040117179, pinned env: mesa 3.3.1, numpy 2.4.6,
  networkx 3.6.1, Python 3.11).
- Analysis tools: `tools/alpha_star_interiority.py`, `tools/compare_configs.py`.
- Manuscript: `DisasterAIFilter_NHB.tex` on branch
  `claude/latex-merge-restructure-0x2rhf`. Two lengths exist in git history:
  ~6,000 words (commit f29a4dd, current) and ~3,000 words (commit 2c2a09e —
  the better base for PNAS). Paper assets (References.bib, Figures/, sn-jnl
  class, Supplementary_S1-S9_content.tex) live in
  `tinacomes/DisasterAIFilterPaper`.

Priorities: Phases 1–3 are the scientific substance (do first); Phase 4 is
manuscript reshaping; Phase 5 is packaging. Steps marked **[HUMAN]** need the
author.

---

## Phase 1 — Individual-confirmation ablation (highest priority)

Purpose: the headline mechanism (consensus recycling) currently rests on the
*assertion* that confirming individual priors instead of the network consensus
would not amplify social echo chambers. Demonstrate it.

**Claude Code prompt:**

```
In tinacomes/DisasterAI, add an ablation switch `--confirmation_target
{consensus,individual}` (default: consensus, current behaviour). In
DisasterAI_Model.py, the AI response path (AIAgent, ~lines 2117-2240) uses
caller.get_network_consensus() as the confirmation target b for exploitative
callers; under `individual`, the AI must confirm the caller's own current
belief for BOTH agent types, leaving everything else unchanged. Thread the
flag through simulate.py and the run-primary-sweep / compare-network-mobility
workflows as an input. Add a smoke test asserting that at alpha=1 with
target=individual the AI report equals the caller's prior. Then trigger the
paired sweep (11 alpha levels, N=20, both configurations, same seeds as run
32040117179) with target=individual, and produce a comparison figure and
table (SECI, AECI-Var, MAE vs alpha; consensus vs individual, main model)
in the style of tools/compare_configs.py. Success criterion to report: does
the seed-robust SECI deepening at alpha>=0.7 in the main model disappear
under individual confirmation? Commit results under results/ablation-
individual-target/ with a provenance README like results/README.md.
```

## Phase 2 — Robustness envelope (SI Appendix material)

**Claude Code prompt (run as separate sessions per sweep if long):**

```
In tinacomes/DisasterAI, extend the experiment infrastructure to run four
robustness sweeps under the main-model configuration (network-gated,
mobility on), N=20 seeds each, reusing the existing workflow pattern:
(1) population scaling: N_agents in {100, 300, 500} x alpha in {0.0, 0.5,
0.7, 0.9, 1.0} — does the SECI departure at alpha>=0.7 and the interior
optimum persist? Scale community count or size explicitly and document the
choice. (2) topology variant: add one alternative generator (Watts-Strogatz
small-world within communities, or a degree-heterogeneous variant) behind
--network_type, same alpha grid. (3) AI supply: number of AI agents in
{1, 5, 10}, same alpha grid. (4) verification probability: p_verify in
{0.1, 0.3, 0.5} x alpha grid — this parameter drives the exploratory-agent
recovery collapse, so report echo-chamber recovery ticks alongside SECI.
For each sweep, commit results + a summary figure to results/robustness-<name>/
with provenance READMEs, and write a one-paragraph summary per sweep into a
new file docs/robustness_summary.md stating whether the qualitative findings
(structural precondition, interior optimum, recovery collapse) hold.
```

## Phase 3 — Statistics upgrade and dyadic docking

**Claude Code prompt (statistics):**

```
In tinacomes/DisasterAI, add tools/sweep_regression.py: load the archived
per-replication steady-state results (results/baseline-sweep and
results/comparison, see experiment_results.json / summary tables), and fit,
for each outcome metric (SECI, AECI-Var, AECI-Err, MAE, unmet needs,
precision), a regression with alpha (continuous + quadratic), configuration,
and their interaction, with seed as grouping factor (mixed model via
statsmodels MixedLM, or seed-clustered OLS if MixedLM is unstable). Report
standardized effect sizes and Holm-corrected p-values for the per-level
paired contrasts currently reported as raw 95% CIs. Also re-run the alpha in
{0.8, 0.9, 1.0} cells of the primary paired sweep with N=50 seeds (extend
seed_base) to firm up the boundary claims, archived under
results/boundary-n50/. Output a markdown table suitable for the SI.
```

**Claude Code prompt (dyadic docking):**

```
In tinacomes/DisasterAI, add a minimal two-agent docking experiment: one
human agent (each type in turn) and one AI agent, no social network, no
relief (or relief disabled), sweeping alpha. Measure whether repeated
human-AI interaction amplifies the human's initial belief bias relative to
a human-human baseline pair, qualitatively reproducing the dyadic
amplification pattern reported by Glickman & Sharot (glickman2024human in
References.bib of tinacomes/DisasterAIFilterPaper). Keep it small
(new script experiments/dyadic_docking.py + one figure). The goal is a
one-paragraph SI section: 'the model recovers the published dyadic effect;
the paper shows what changes in networks.'
```

## Phase 4 — Manuscript reshaping to PNAS

**Claude Code prompt:**

```
Create the PNAS version of the manuscript. Start from the ~3,000-word
version of DisasterAIFilter_NHB.tex at commit 2c2a09e on branch
claude/latex-merge-restructure-0x2rhf in tinacomes/DisasterAI (git show
2c2a09e:DisasterAIFilter_NHB.tex). Produce DisasterAIFilter_PNAS.tex:
(a) main text ~4,000-4,500 words; 4 main display items: Fig 1 concept
(TikZ), Fig 2 configuration comparison, Fig 3 a composed Goldilocks figure
(alignment sweep + cognitive-gap sweep as panels a/b), Fig 4 periphery.
Everything else (lifecycle, alpha* sensitivity, AI query share, paired-
deltas table, metrics table if needed) moves to the SI Appendix.
(b) Abstract <=250 words; add a Significance Statement <=120 words written
for a general scientific audience. (c) Convert spelling to American
English. (d) Keep the four-gaps -> four-findings structure, the extreme-
case framing of disasters, and the no-figuresplaining style ('The result
is X (Fig. 2)', never 'Fig. 2 shows'). (e) Add a short paragraph in the
Discussion making explicit that the interior alignment optimum is a
descriptive property of the trade-off, not a recommendation to build
deliberately confirming AI, and name adoption routes that do not require
confirmation (uncertainty communication, transparency, verification
support). (f) Do not add references; all keys must resolve against
References.bib in tinacomes/DisasterAIFilterPaper. Note in a header
comment that the class file should be swapped to the official PNAS
template (pnas-new.cls) at submission; keep sn-jnl compilable meanwhile.
```

## Phase 5 — SI Appendix, calibration table, packaging

**Claude Code prompt:**

```
Build the PNAS SI Appendix as SI_Appendix_PNAS.tex, starting from
Supplementary_S1-S9_content.tex in tinacomes/DisasterAIFilterPaper plus the
material moved out of the main text in Phase 4 and the new results from
Phases 1-3. Add: (a) an ODD-protocol description of the model (Overview,
Design concepts, Details — derive from the Methods and DisasterAI_Model.py);
(b) a calibration table mapping every structural parameter (within-community
edge probability, bridge probability, community count, home radius, return
bias, explorer behaviour) to the published empirical statistic or source
that motivates it (bruns2012qldfloods, sutton2008backchannels,
pappalardo2015returners, granovetter1973strength) and flagging which values
are stylised choices; (c) the ablation, robustness, regression, and dyadic
docking sections with their figures; (d) the promises/perils table (S10).
Keep all existing S1-S9 numbering stable where possible.
```

**[HUMAN] remaining steps:**

1. Resolve the flagged placeholder reference `steinbrink2024transparency` in
   References.bib, and decide on the removed "Fu et al. 2023" citation
   (TODO comment in Methods).
2. Mint a Zenodo DOI for the DisasterAI repo (code + results/) and add it to
   Data/Code availability. (Claude Code can prepare the .zenodo.json.)
3. Approve the Significance Statement wording; complete Acknowledgements and
   Funding.
4. Cover letter: lead with "extends the dyadic human-AI feedback-loop
   findings (Glickman & Sharot, NHB 2024) to networks; closest methodological
   ancestors Lorenz 2011 / Becker 2017 (both PNAS)". Suggest 3-5 reviewers
   from computational social science / collective intelligence / crisis
   informatics.
5. Submit via the PNAS portal (Direct Submission).
