# Submission plan: Nature Human Behaviour (alternative route)

Target: NHB Article. **Only choose this route if you are willing to add an
empirical component** — realistically an online experiment, since no network
dataset is available for calibration. Without it, expect a substantial
desk-reject risk: NHB accepts computational modelling, but simulation-only
papers are rare there and usually arrive coupled to data or experiments.

The asset that makes NHB worth attempting at all: the paper directly extends
Glickman & Sharot's dyadic human–AI feedback-loop findings — published in
NHB — to the network level. The cover letter and framing must lead with this.

## Repo map (context for every Claude Code session)

Identical to SUBMISSION_PLAN_PNAS.md — read its "Repo map" section first.
Manuscript base for NHB: the current ~6,000-word `DisasterAIFilter_NHB.tex`
(commit f29a4dd, branch `claude/latex-merge-restructure-0x2rhf`), which is
already in NHB shape (Intro ~1,700 / Results ~2,950 / Discussion ~1,300,
7 results figures + concept figure, Methods unlimited).

## Shared scientific work

Phases 1–3 and Phase 5 of SUBMISSION_PLAN_PNAS.md apply unchanged and should
be completed for NHB too (ablation, robustness envelope, statistics upgrade,
dyadic docking, ODD protocol, calibration-by-literature table). Run those
prompts as written. The steps below are NHB-specific and come on top.

---

## Phase N1 — Empirical anchor without new data (minimum viable)

Do all three; together they are the floor for an NHB attempt.

**(a) Calibration-by-published-statistics** — covered by the PNAS Phase 5
calibration table, but for NHB promote it: adjust the network generator
defaults to *reproduce* published statistics rather than merely citing them.

**Claude Code prompt:**

```
In tinacomes/DisasterAI, extract the published network statistics from the
sources already cited in the manuscript (bruns2012qldfloods: Queensland
floods retweet network; sutton2008backchannels; pappalardo2015returners:
returner/explorer mobility parameters). For each statistic that maps onto a
generator or mobility parameter (mean degree, modularity / community
structure, share of bridging ties, home-radius distribution), compare the
model's current generated networks (network_type used by the main model)
against the published value, in a short script tools/network_calibration.py
that outputs a comparison table. Where the generated statistic is far off,
propose (do not silently apply) adjusted generator parameters, and quantify
via a small sweep whether the headline findings are insensitive to the
adjustment. Deliverable: docs/calibration_report.md + the table for the
supplement, phrased as 'generator parameters reproduce published disaster
communication network statistics'.
```

**(b) Pattern-oriented validation against the author's own field findings.**

**Claude Code prompt:**

```
In tinacomes/DisasterAI, add a model-validation analysis showing that the
alpha=0 baseline of the main model reproduces three empirically documented
qualitative patterns, before any AI-alignment claims: (1) informational
silo formation (pan2012crisis), (2) local-focus misallocation — relief
concentrated near responders' locations at the expense of distant high-need
cells (wolbers2018introducing, sigala2022mitigating), (3) periphery
information deficits — belief error increasing with distance from events
(comes2020coordination). Compute each pattern from the archived alpha=0
replications in results/ (re-run with extra collectors only if a needed
quantity is not logged). Produce one three-panel figure + a half-page LaTeX
subsection 'Model validation' for the manuscript, framed in the
pattern-oriented modelling tradition (cite epstein2006generative — do NOT
add new references; Grimm/Railsback may be described without citation).
```

**(c) Dyadic docking** — as in PNAS Phase 3; for NHB this is near-mandatory
because it ties the model to the NHB-published dyadic result.

## Phase N2 — Online experiment [HUMAN-led; transforms the submission]

The design that tests the model's core prediction, in the Becker/Centola
tradition: networked groups of ~16-20 participants estimate a quantity with
ground truth (e.g., damage severity from imagery) over rounds; an "AI
advisor" gives either truthful or belief-confirming advice (confirmation
implemented as weighted echo of the participant's or the group's prior);
2x2: advisor policy x communication structure (networked vs unrestricted).
Prediction from the model: confirming advice deepens within-group belief
convergence and error only in the networked condition; truthful advice
produces advisor-reliant convergence in both.

- **[HUMAN]**: ethics approval (TU Delft HREC), preregistration (OSF),
  Prolific budget (~200-400 participants), decide on incentive-compatible
  scoring.
- **Claude Code can build**: the experiment software (oTree or Empirica
  implementation of the design above), power analysis using effect sizes
  simulated from the model itself, preregistration draft, and the analysis
  pipeline mirroring tools/sweep_regression.py.

**Claude Code prompt (software):**

```
Create a new repo/directory experiment/ implementing the networked
estimation experiment described in SUBMISSION_PLAN_NHB.md Phase N2 in
oTree: rounds of private estimation of a ground-truth quantity, visibility
of neighbours' estimates according to an assigned network (ring-lattice
communities with bridge ties vs full graph), and an AI-advisor message per
round whose content is truthful or a confirming echo per treatment.
Include: config for the 2x2 design, data export matching the model's
outcome metrics (within-group belief variance vs population, error,
advisor-reliance), a bot-based smoke test, and a power analysis notebook
that draws effect sizes from simulating the model at matched parameters.
```

## Phase N3 — Manuscript and NHB packaging

**Claude Code prompt:**

```
Update DisasterAIFilter_NHB.tex on branch claude/latex-merge-restructure-
0x2rhf in tinacomes/DisasterAI for NHB submission: (a) integrate the new
results: ablation (individual vs consensus confirmation), robustness
summary sentence(s), the model-validation subsection from the pattern-
oriented analysis, dyadic docking paragraph, and — if Phase N2 ran — the
experiment as a Results subsection with the model-experiment comparison.
(b) Add the ethics paragraph on interpreting the interior optimum (see
PNAS Phase 4 item e). (c) Keep main text <=6,000 words (currently ~5,975 —
compensate any additions with equivalent trims, protecting the four-gaps ->
four-findings structure and the extreme-case framing). (d) Keep British
spelling (NHB accepts both; the file is currently British). (e) No new
references; every key must resolve against References.bib in
tinacomes/DisasterAIFilterPaper. (f) Update the supplementary-information
backmatter listing for any figures added or moved.
```

**[HUMAN] remaining steps:**

1. NHB requires a Reporting Summary and (if Phase N2 ran) ethics statements,
   preregistration links, and participant-consent documentation.
2. Resolve `steinbrink2024transparency` placeholder and the "Fu et al. 2023"
   TODO; complete Acknowledgements/Funding; Zenodo DOI (as in PNAS plan).
3. Cover letter: lead with the Glickman & Sharot (NHB) continuation, then
   the structural-precondition finding; state explicitly what is model,
   what is validation, what is experiment.
4. Decision point: if Phase N2 is not feasible within your timeline, submit
   to PNAS instead (SUBMISSION_PLAN_PNAS.md) — the shared scientific work
   (Phases 1-3, 5) transfers without loss.
