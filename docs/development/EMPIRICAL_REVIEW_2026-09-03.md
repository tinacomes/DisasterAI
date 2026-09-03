# Empirical-foundations and venue review (2026-09-03)

Review of `PAPER_BRIEF.md` (*Empirical foundations* and *Target venue*
sections) and `docs/EMPIRICAL_FOUNDATIONS_PLAN.md` against the E1
implementation (`experiments/docking_fit/`, `results-docking-fit/`) and
the CI workflows, plus a re-decision of the venue under the APC
constraint the author established (TU Delft and DLR library support
each cap at ≈ USD 2,000 per article; nothing above that can be paid).
Internal working material — `docs/development/` is deleted before the
public archive.

Method: full read of both planning documents and the E1 code and
results; `fit_docking.py --report-only` re-run (regenerates
`fit_report.md`/`docking_fit.png` byte-identical to the committed
versions); `population_check.py` smoke-run; APCs and institutional
agreements checked on 2026-09-03 via web search (publisher and TU Delft
pages themselves were not fetchable from this session — the "verify"
flags below are for the library's Journal Browser).

## 1. Verdict

1. **The two documents are consistent with each other** on the E2
   decision (parked, E2-lite in its place), the E1 status, and the
   execution order. Seven residual inconsistencies/overclaims are
   listed in §3 and fixed in this pass where they are documentation.
2. **E1 is not complete.** The *fit* is executed and reproducible; the
   *deliverable claim* (plan §2 step 4: the canonical N=20, 11-α,
   two-configuration population sweep at the fitted parameters) has
   not been run, and until it has, "micro-parameters empirically
   estimated, population results intact" rests on a 3-α × N=5 reduced
   grid. The missing CI plumbing is added in this pass (`--param-
   overrides` in `test_filter_bubbles.py`; workflow
   `run-docking-fit-sweep.yml`); the run, the archive, and the SI
   fold-in remain (§2, §5).
3. **The fit's own wording overclaims in three places** (§2, F2). The
   honest reading is *weaker* than the report's, though still useful:
   the human–human target is uninformative (CI spans −0.6 to 1.3), the
   model under-transmits AI influence for both types and by
   construction for confirmation-seekers, and two fitted parameters sit
   at the search-box boundary.
4. **Venue: Science Advances is out on cost, and so are Nature
   Communications and any Nature-portfolio gold-OA route.** The
   recommendation becomes a zero-APC strategy with three tiers
   (§4): Nature Human Behaviour via the subscription route (reach),
   Computers in Human Behavior via the Dutch Elsevier / DEAL agreements
   (best scope match, safe), Journal of the Royal Society Interface via
   Subscribe-to-Open 2026 (modelling audience). All three carry both
   agent-based modelling and human–AI interaction; none is a
   computer-science venue.

## 2. E1 — what exists, what is missing, what is overclaimed

**What exists and checks out.**

- `compute_targets.py` computes the targets from Glickman & Sharot's
  public per-trial data with the authors' own trial/block conventions;
  the source commit is recorded in `targets.json`. This is better than
  the plan's "extract from the paper" and should be said in the SI.
- `fit_docking.py` (96-point Latin hypercube, 8 coarse seeds, top-8
  refinement at 20 seeds on five α levels) runs the M6 harness
  unchanged at defaults; the report regenerates identically from the
  committed CSV/JSON (verified today via `--report-only`).
- `population_check.py` runs the population model at the fitted
  overrides on a reduced grid and its four structural checks pass
  (`population_check_fitted.json`, `all_pass: true`).

**Findings (F) and required actions (A).**

- **F1 — Deliverable not delivered.** Plan §2 step 4 and the brief's E1
  bullet both promise the canonical sweep; `population_check_fitted.json`
  is explicitly "reduced-grid qualitative structure check". There was
  no way to run the canonical sweep at the fitted parameters on CI:
  `test_filter_bubbles.py` exposed no flag for `d_*`/`delta_*`/
  `initial_*_trust`.
  **A1 (done in this pass):** `--param-overrides JSON` added to
  `test_filter_bubbles.py` (accepts a flat dict or the E1
  `fitted_params.json` directly; drops the dyad-only `rounds`; the
  overrides are recorded in every per-α output JSON); workflow
  `.github/workflows/run-docking-fit-sweep.yml` mirrors the
  results-mechfix workflow (same matrix, seeding, collect and compare
  jobs) with the overrides applied. Smoke-tested at 10 ticks.
  **A2 (to do):** trigger the workflow (N=20, 200 ticks), archive under
  `results-docking-fit-sweep/` with a README in the results-* format
  (run id, commit, verdict), and add the like-for-like table
  (fitted vs `results-mechfix/` default) to the E1 report. Seeds are
  index-paired with results-mechfix but RNG consumption differs once
  acceptance decisions differ, so report fitted-vs-default as
  independent samples with CIs, not as paired per-seed deltas, unless
  bit-pairing is verified.
  **A3 (cheap, to do):** run `population_check.py --param-set default`
  so the reduced-grid comparison is like-for-like (only the fitted
  file exists).
- **F2 — Overclaims in `fit_report.md` and the plan's §2 summary.**
  (i) "Human–human transmission is matched": the measured κ_human is
  0.309 with 95% CI [−0.593, 1.343] — any value the model could produce
  is "inside it"; the target carries almost no weight in the loss
  (se 0.49 vs 0.12 for κ_AI). Say: *consistent with an uninformative
  measurement*. (ii) "At the edge of the measured CI": the
  accuracy-seeker's κ_AI = 0.49 is *below* the CI lower bound (0.52),
  and the population mean (0.27) is far below it; no parameter set in
  the box reaches the CI. Say: *below the measured range for both
  types; the direction of the residual mismatch is conservative*.
  (iii) "The mismatch is conservative" needs its consequence spelled
  out once: if anything the population-scale harms are understated,
  because the human model adopts AI judgements more reluctantly than
  measured participants — and the confirmation-seeker's near-zero AI
  transmission is a *structural* property of the D/δ acceptance window
  (cf. C12), i.e. a model limitation the paper must list, not a
  finding. **A4 (done in this pass):** the three interpretation bullets
  in `write_report()` are rewritten (so the report stays regenerable;
  `fit_report.md` regenerated, figure unchanged) and mirrored in the
  plan §2; the SI §M6 wording follows at redraft.
- **F3 — Boundary-constrained fit.** Fitted `rounds` = 60 is the upper
  bound of its box [20, 60] and `d_explor` = 5.37 sits near its bound
  (5.5); the coarse top-10 spans the whole `initial_trust` range
  (0.19–0.60), so that coordinate is not identified. The box is the
  S9/M5-certified envelope, which is a legitimate reason to constrain
  the fit — but the report must say that the optimum is on the
  envelope boundary in two coordinates and that the fit therefore
  reports an *envelope-constrained* best point, not an interior
  optimum. **A5 (done in this pass):** `boundary_note()` in
  `fit_docking.py` now names every coordinate within 10% of a box edge
  in the identifiability section. Optional: one probe run with `rounds`
  up to 100 to show whether κ_AI keeps rising (it should, since
  transmission accumulates per round).
- **F4 — Effectively a one-target fit with seven free parameters.**
  With κ_human uninformative and the orderings satisfied everywhere,
  the loss is κ_AI plus penalties; identification comes from one
  number. The pipeline already computes two further informative
  targets from Exp2 that it does not use: the accurate-AI error ratio
  C2 = 0.897 [0.845, 0.953] (model analogue: the focal human's MAE
  under the truthful AI relative to its solo baseline — `run_dyad`
  already records `mae`) and the biased-AI induced bias C3. The
  per-block induced-bias slope in Exp1 (0.021/block) is a natural
  handle on the time-scale mapping that would pin `rounds` instead of
  letting it run to the bound. **A6:** add C2 as a second loss term
  (and, if the block slope is used, the time-scale mapping); rerun the
  fit. This is a half-day of compute and converts "one identified
  direction" into a defensible two-target fit.
- **F5 — Reproducibility gaps.** `requirements.txt` lacks `pandas`
  and `tabulate` (`fit_docking.py` needs both; the report's
  `to_markdown` calls fail without tabulate) — added in this pass.
  The `coarse_summary.csv`/`refined_summary.csv` files the script now
  writes are absent from `results-docking-fit/` (the committed run
  predates them; the fallback reconstruction is exact — verified).
  `results-docking-fit/` had no README with run provenance, unlike
  every other results-* directory — added in this pass. The fit itself
  has no CI workflow (it ran in-session); the README records that.
- **F6 — Plan text.** §1 says "E1 pulled forward and DONE" while §2
  lists follow-ups; the E1 bullet in §1 is still in the future tense;
  §6 lists no workflow or results directory for the sweep. Fixed in
  this pass.

**Remaining E1 work, in order:** A6 (rerun the fit with C2; ~0.5 d) →
A2 (canonical sweep on CI; ~1 h wall clock, then archive + table;
0.5 d) → A3 → SI §M6 fold-in at redraft. Doing A2 before A6 would mean
running the expensive sweep twice.

## 3. Documentation consistency (brief vs plan)

Fixed in this pass unless marked *redraft*:

1. Brief E1 bullet claimed the canonical sweep was "queued for CI" —
   no workflow existed. Now points to the workflow and states the
   deliverable as pending.
2. Brief *Target venue* and *Venue coupling*, plan §5, and the
   skeleton preamble all read Science Advances — superseded (§4).
3. Plan §1 execution order "DONE" vs §2 follow-ups — reconciled.
4. Plan §2 status summary carried the F2 overclaims — reworded.
5. Brief *Limitations to state* lacked the E1 structural finding
   (confirmation-seekers cannot reproduce the measured AI transmission
   by construction) — added.
6. Brief evidence-map docking row cites only `results-docking/` —
   now also `results-docking-fit/`.
7. *Redraft:* SI §M6 still says "not parameter-matched calibration";
   `results-docking/README.md` likewise. Both are correct for M6 as
   archived and become stale only when the E1 section is folded in;
   leave until then.

E2-lite gains a first-rank anchor that did not exist when the plan was
written: Cheng et al., *Sycophantic AI decreases prosocial intentions
and promotes dependence*, Science 391, eaec8352 (2026) — 11 deployed
models affirm users 49% more often than humans do; users prefer and
trust the sycophantic responses and return to them. That is the
capture mechanism (reliance up, judgement worse) measured in people.
Verify against the publisher record and add to `References.bib` in the
E3 pass; cite it in the abstract-level framing for any venue (§4).

## 4. Venue re-decision under the APC constraint

**Constraint.** Library support: TU Delft ≈ USD 2,000 and DLR ≈ USD
2,000 per article, and nothing beyond. Facts checked 2026-09-03:

| Journal | Route | List price | Net to author | Status |
|---|---|---|---|---|
| Science Advances | gold OA only | USD 5,450 | ≈ USD 3,450 | **out** |
| Nature Communications | gold OA only | USD 7,350 (£5,490) | ≈ USD 5,350 | **out** |
| Nature Human Behaviour | gold OA option | USD 12,850 | ≈ USD 10,850 | out |
| Nature Human Behaviour | **subscription route** | none | **0** | in |
| Nature Machine Intelligence | subscription route | none | 0 | in on cost; scope stretch (brief) — not recommended |
| Computers in Human Behavior (Elsevier, hybrid) | OA under the Dutch Elsevier agreement (TU Delft corresponding author; valid to 31 Dec 2026) or DEAL (DLR corresponding author, 2024–2028) | covered | **0** | in |
| J. R. Soc. Interface / Proc. R. Soc. A | Subscribe-to-Open 2026: OA, CC-BY, no APC | none | **0** | in (2026 acceptances; 2027 continuation to be confirmed) |
| Royal Society Open Science | born-OA; APC unless a Royal Society Read & Publish covers TU Delft | APC | verify | conditional |
| J. Computational Social Science (Springer, hybrid) | Dutch Springer agreement (2,067 articles/yr quota; ran out in autumn 2024 and 2025) or DEAL | covered | 0 | author-excluded on audience grounds |
| PNAS | delayed OA | USD 2,575 | ≈ USD 575 | affordable, author-excluded |

Notes: the Dutch Springer Nature agreement covers Springer *OpenChoice*
hybrids, not Nature-branded journals; DEAL likewise excludes
Nature-branded journals. NHB's subscription route is free at
submission, and Dutch law (Taverne, TU Delft "You share, we take care")
puts the version of record in the repository six months after
publication at no cost; an arXiv preprint at submission covers
immediate access, as the brief already prescribes.

**Recommendation (three tiers, all APC-free, all covering ABM and
human–AI interaction, none pure CS):**

1. **Nature Human Behaviour — subscription route (reach).** Fit: the
   docking target is an NHB paper; NHB's own agenda pieces (*Machine
   culture*, 2023; *A new sociology of humans and machines*, 2024)
   call for exactly this genre — human–machine social systems studied
   with computational models. Risk: the modal outcome for a
   simulation-only research article is desk rejection; the cost of
   finding out is small. Do a presubmission enquiry with the
   machine-behaviour framing and the E1 fit; if declined, go to tier 2
   without redrafting the content.
2. **Computers in Human Behavior (safe, best scope match).** Scope is
   "the psychological impact of computer use on individuals, groups
   and society", computers "only as a medium through which human
   behaviours are shaped" — the brief's "behaviour first, model as
   instrument" rule word for word. ABM/simulation studies of echo
   chambers and social-media dynamics are published there; IF 12.2
   (2025 data). Zero cost via the Dutch Elsevier agreement or DEAL.
   Risk: psychology reviewers asking for human data — answered by the
   E1 fit (micro-parameters from a published experiment), the E4
   predictions table, and the Cheng et al. anchor (capture measured in
   people). Format: no hard length cap; Introduction–Method–Results–
   Discussion, so the roomy skeleton survives with Methods moved
   forward.
3. **Journal of the Royal Society Interface (modelling audience).**
   ABMs of opinion dynamics with social-identity bias, collective
   wisdom under incorrect social information, and misinformation on
   networks are all in its recent record; the Lorenz/Becker lineage
   argument lands. APC-free for 2026 acceptances under
   Subscribe-to-Open. Risk: narrower AI readership; arXiv carries that
   audience.

Fallbacks, all covered by the same agreements: International Journal
of Human-Computer Studies (Elsevier), Technological Forecasting &
Social Change (Elsevier; sociotechnical ABMs are common there), Royal
Society Open Science (if a Royal Society Read & Publish is in place —
verify). JCSS only if the author lifts the audience exclusion.

**Framing that travels across the three tiers.** The brief's framing
rules stay binding; two additions:

- Lead every version with *people*: who relies on what, who learns
  what from whom, and why confirmation starves accuracy-seekers and
  captures confirmation-seekers. The AI is a behavioural dose, never
  an alignment-engineering object; no RLHF/preference-optimisation
  machinery in the abstract, and the title names human behaviour.
- Anchor the abstract's motivation in the measured human evidence
  (Glickman & Sharot 2025 for the dyad; Cheng et al. 2026 for
  sycophancy → dependence) and state the paper's move in one sentence:
  the dyadic effects are measured, the population-scale consequences
  are not, and the model is the instrument that makes them measurable.
  For CHB, add one sentence on reliance-without-trust as the
  psychological construct the model operationalises.

**Decision rule.** Presubmission enquiry to NHB at the end of the
redraft; on a decline or no reply within three weeks, submit to CHB.
Interface is the fallback if CHB rejects on "no human data". Post to
arXiv (cs.CY + cs.MA, cross-list physics.soc-ph) at first submission
regardless. Verify the Elsevier agreement's 2027 renewal and the
Royal Society S2O status if the timeline slips past December 2026.

## 5. Action list

| # | Action | Owner / effort | Status |
|---|---|---|---|
| A1 | `--param-overrides` flag + `run-docking-fit-sweep.yml` | done (this pass) | ✓ |
| A6 | Add Exp2 C2 (and optionally block-slope time scale) to the loss; rerun fit | 0.5 d compute | open |
| A4/A5 | Reword report interpretation and identifiability (boundary, one informative target) | done (this pass) | ✓ |
| A2 | Trigger the sweep (N=20), archive `results-docking-fit-sweep/` + README + fitted-vs-default table | 1 h CI + 0.5 d | open |
| A3 | `population_check.py --param-set default` | 10 min CPU | open |
| — | SI §M6 fold-in (text + Fig. S6 panel from `docking_fit.png`) | redraft | open |
| — | Presubmission enquiry to NHB; CHB as the standing target | author | open |
| — | E4 → E3 (+ E2-lite with the Cheng et al. anchor) | as planned | open |
