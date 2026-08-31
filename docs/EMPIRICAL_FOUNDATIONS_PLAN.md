# Tier-1 empirical foundations — verification and implementation plan

Companion to `PAPER_BRIEF.md` §*Empirical foundations*. That section
lists the upgrade path (E1–E8) and its rationale; this document (i)
re-verifies, as of **2026-08-31**, that the Tier-1 steps E1–E4 are the
right pre-submission investments, (ii) makes E1, E3, and E4 concrete
enough to execute, and (iii) records the venue re-confirmation. The
full E2 protocol and implementation plan is
`docs/ALPHA_MEASUREMENT_PROTOCOL.md`.

## 1. Verification: are E1–E4 the right steps? (checked 2026-08-31)

Verdict (revised 2026-08-31 after the author's scope decision):
**E1/E3/E4 yes, with one rescope (E1); E2 is PARKED as a standalone
side project and replaced in-paper by E2-lite (§2a).** The check was
against (a) what the paper's referee model actually needs — the
brief's four-layer robustness answer covers internal validity; Tier 1
exists to answer "your α is a cartoon" with external anchors — and
(b) the 2025–26 literature.

- **E2 (α measurement of deployed systems) — scientifically
  confirmed, editorially parked (author decision 2026-08-31).** The
  science holds: as of the check date no published work estimates a
  truth–belief mixing weight against controlled in-context ground
  truth, and the field is hot enough that the novelty window is
  closing. But it is a *second study* — its honest in-paper cost is a
  Results subsection, a Methods block, figure real estate, and a
  paragraph of validity caveats, which the paper's length budget
  cannot carry at any of the three candidate venues without displacing
  the credibility chain the redraft exists to foreground. Decision:
  E2 proceeds on its own track as its own (short) paper — the full
  plan stays frozen in `docs/ALPHA_MEASUREMENT_PROTOCOL.md`, executed
  when the author chooses, with preregistration protecting priority.
  The two papers then cross-cite: the measurement paper supplies the
  "you are here"; this paper supplies the population-scale
  consequences. In this paper, E2 survives as one sentence in the
  Discussion research agenda — and as a companion-preprint citation
  if the side project lands before submission.
- **E4 (predictions table) — confirmed, trivially cheap.** The
  literature now contains dyadic and individual-level evidence of
  dependence and reliance effects; a table of the model's *networked*
  signatures (starvation concentrated on accuracy-seekers, reliance-
  without-trust capture, query-diversity narrowing, repair lag) is
  exactly what makes the paper a hypothesis generator for that active
  field. One day of work; no dependency on anything.
- **E3 (calibration-table citations) — confirmed.** Table S7
  (`PNAS_Paper/SI_Appendix.tex`, tab:s7-params) currently carries
  empirical sources for only two rows (Pappalardo mobility;
  Granovetter weak ties); every other row is design rationale.
  Changes no result, changes reviewer priors; days of work.
- **E1 (parameter-matched docking) — confirmed but rescoped.**
  Glickman & Sharot's tasks (perceptual dot estimation, emotion
  aggregation, social judgement) do not share units with the 0–5
  severity grid, so literal parameter matching is not on the table.
  The honest, achievable upgrade is **effect-size-targeted fitting**:
  fit the dyad's free parameters to dimensionless targets from their
  published results (amplification ratios, convergence direction and
  relative speed, AI-vs-human asymmetry). The brief's phrase
  "parameter-matched" should be read, and eventually written, as
  "effect-size-matched"; "micro-parameters empirically estimated"
  survives with that qualifier. Feasibility gate: their data/code
  availability (the NHB paper carries availability statements —
  verify at kickoff; fallback is fitting to reported summary
  statistics only, which still supports the claim).

Nothing in the check argues for promoting a Tier-2 step into Tier 1:
E5/E6 remain confirmatory-flavoured and slow (E6 — rerunning on an
empirical network — is the natural next candidate *if* the author
wants one more empirical layer, but it is weeks of compute and reads
as confirmation, not as a new anchor).

**What "empirically stronger" costs in paper length** — the decisive
axis after the E2 decision. Every remaining step is length-cheap:

| Step | Main-text cost | Where the substance lives |
|---|---|---|
| E1 (fitted docking) | ~1 Methods sentence + 1 Results clause | SI section + SI figure |
| E2-lite (literature anchor) | ~120 words, Discussion | References.bib |
| E3 (calibration citations) | 0 | SI Table S7 |
| E4 (predictions table) | 1 Discussion table + ~150 words | traces to existing H-claims |

With E2 parked, **E1 is the paper's single biggest remaining
empirical upgrade**: it converts the docking anchor from "reproduces
the published pattern" to "micro-parameters estimated from the
published experiment, population results intact" — indirect
calibration — at near-zero length cost.

**Execution order (revised 2026-08-31):**

    E4 (1 d) → E3 (2–4 d) → E2-lite (0.5 d, folds into E3's
    literature pass) → E1 (1–2 wks; the long pole)

    E2 side project: independent track, scheduled by the author;
    its only coupling to the paper is an optional companion citation.

E4 and E3 first because they are self-contained, feed the Discussion
and SI directly, and lose nothing by preceding the others. E1's
headline ("fitted vs. default parameters leave the population results
intact") is a robustness statement that lands the same whenever it is
done — but it must be finished before submission, so it starts as
soon as the cheap steps clear.

## 2. E1 — Effect-size-targeted docking (implementation)

**Status (2026-08-31): EXECUTED** — pipeline in
`experiments/docking_fit/` (targets → fit → population check), results
in `results-docking-fit/` (`fit_report.md` carries the SI-ready
interpretation). Better than planned: Glickman & Sharot's per-trial
data is public (affective-brain-lab/BiasedHumanAI), so the targets are
*computed from the primary data* with the authors' own conventions,
not transcribed from the paper. Headline results: the human–human
transmission coefficient is matched (~0.28 vs measured 0.309); the
identified parameter direction is the initial AI trust (≈2× the
default); the accuracy-seeker's AI transmission reaches the edge of
the measured CI (0.49 vs [0.52, 0.98]) while the confirmation-seeker
stays near zero at every certified parameter set — the model's own
D/δ acceptance mechanism (cf. C12) — so the model *under*-transmits AI
influence relative to measurement: a conservative mismatch, stated as
such. All orderings (AI retains bias > human partner; monotone in α)
reproduce. Remaining: the canonical N=20 full-grid population sweep at
fitted parameters on CI (the in-session check runs the reduced grid).
The plan below is retained for provenance.

Harness: `experiments/dyadic_docking.py` (exists, runs the dyad with
the model's own acceptance/trust machinery; currently qualitative).

1. **Targets** (dimensionless, from Glickman & Sharot's published
   results; extracted at kickoff and frozen in
   `experiments/docking_fit/targets.json` with the quote/table each
   number comes from): (i) bias amplification of the human×AI loop
   relative to human×human interaction; (ii) monotone increase of
   final bias in the AI's alignment to the user; (iii) asymmetry:
   AI-induced bias persists/grows where human-partner bias decays.
   Each target is a ratio or an ordering, never a raw unit.
2. **Free parameters:** acceptance D, δ (both types), trust learning
   rates (`exploit_trust_lr`, `explor_trust_lr`), rounds. Everything
   else pinned to Table S7.
3. **Fit:** simulated minimum distance — Latin-hypercube or coarse
   grid over the 6-D box (bounds: the S9 profile-sweep ranges, which
   the robustness envelope already certifies), loss = weighted
   distance on the target ratios, ≥ 20 seeds per point (reuse the
   docking script's seed loop); report the fitted set with a
   profile/identifiability note (expect ridges — say so rather than
   overclaiming point identification). New code:
   `experiments/docking_fit/fit_docking.py` wrapping `run_dyad`.
4. **The deliverable claim:** rerun the *population* main
   configuration at the fitted parameter set (one 11-α × 20-seed
   sweep) and show the headline structure (interior α\*, starvation
   gradient, structural precondition) intact → one SI section + one
   sentence in Methods ("dyadic micro-parameters estimated from
   published human–AI effect sizes; population results unchanged"),
   upgrading M6 from consistency check to indirect calibration.
5. **Effort:** 1–2 weeks incl. the sweep; compute is the binding
   resource (reuse the CI parallel-run tooling used for mechfix).

## 2a. E2-lite — literature-anchored α plausibility (implementation)

The in-paper replacement for the parked E2: one Discussion paragraph
(~120 words) arguing from *published* measurements that the interior
of the α axis is the empirically relevant region. Structure of the
argument, each clause carrying a verified citation:

1. Deployed assistants are measurably sycophantic but not fully
   confirming — benchmark studies find systematic but partial
   agreement drift (Sharma et al.'s SycophancyEval line; the 2025–26
   multi-turn benchmarks) → real systems sit strictly inside (0, 1),
   not at either endpoint the paper's extremes represent.
2. The confirmation weight is not fixed: it rises under user pushback
   and sustained pressure (turn-of-flip / number-of-flips results) →
   a *range* on the axis, moving in the direction the paper's
   high-dose results describe.
3. Therefore the dose–response interior — where starvation, capture,
   and the operational optimum live — is where deployed systems
   plausibly operate; precisely locating them is the stated agenda
   (the parked E2, one sentence, alongside E7/E8).

Binding phrasing rules: never convert a published benchmark score
into an α̂ value (different estimands — stance flips are not mixing
weights); never name a specific system as "α ≈ x"; the paragraph
claims *region and direction*, not location. Deliverable: the
paragraph + 3–5 verified References.bib entries (folds into E3's
literature pass). Effort: ~0.5 day.

## 3. E3 — Calibration-table upgrade (implementation)

For each Table S7 row now justified by design rationale, add the
empirical literature that estimates or bounds it. Working list of
citation targets (all to be located, verified, and entered into
`PNAS_Paper/References.bib` per the no-memory-citations rule):

| Row(s) | Literature to source |
|---|---|
| Acceptance D, δ (both types) | Bounded-confidence window estimates from opinion-dynamics experiments/calibrations (Deffuant/HK empirical literature) |
| Trust learning rates; initial trust | Human–automation trust formation/updating dynamics (trust-in-automation literature, incl. trust-repair rates) |
| Verification lag/probability; report expiry | Situation-report cycle times and information-verification lags in crisis-coordination / crisis-informatics literature |
| Relief outcome delay 15–25 ticks | Humanitarian logistics lead-time literature |
| Q-learning rate / ε | Standard RL parameter ranges in cognitive-modelling applications |
| Share exploitative 0.5 | Framed as agnostic default + the 132-cell sweep; cite individual-difference evidence for confirmation-seeking heterogeneity |
| Rumor parameters | Crisis rumoring literature |

Method: one pass to collect candidate sources; verify each against the
publisher record; add a fourth column "Empirical anchor" to Table S7
(or fold into the Rationale column) with the citation and, where the
literature gives a number, the mapping sentence ("a tick ≈ …, so 15–25
ticks ≈ …"). Where the honest answer is "no literature estimates
this", keep the design rationale and say so — a mixed column is more
credible than a uniformly decorated one. Deliverable: updated
SI_Appendix.tex Table S7 + References.bib entries. Effort: 2–4 days.

## 4. E4 — Predictions table (implementation)

One Discussion table, drafted directly from the H-claims; each row =
falsifiable signature, observable, and the data stream it is visible
in. Draft rows (numbers stay in the compendium; the table states
directions and loci):

| Model signature (source claim) | Observable prediction | Where visible |
|---|---|---|
| Starvation of accuracy-seekers (H4) | Accuracy costs of sycophantic assistants concentrate on users who seek accuracy, via shrinking diversity of consulted sources, not via persuasion | Panel studies pairing usage logs with belief elicitation |
| Capture without trust (H5) | Reliance (query share) rises with assistant confirmingness while self-reported trust stays flat | Assistant usage logs + trust surveys |
| Query-diversity narrowing (H4/H5) | Topical/source diversity of an individual's queries declines with exposure to confirming responses | Chat/search logs |
| Structural precondition (H3) | Echo-chamber deepening from assistant use appears in network-bounded information environments, not open ones | Cross-platform comparisons; bounded-visibility field settings |
| Front-loaded harm + repair lag (H9) | Fixing a sycophantic model dissolves conformity quickly, but belief-pool depletion persists; late-adopting populations are buffered | Longitudinal data around model updates (e.g. deployed-model rollbacks) |
| Periphery concentration (H8) | Harms concentrate on spatially/socially peripheral users | Geographically resolved outcome data in crisis response |

Rules: every row traces to a compendium H-item; phrase as predictions
("the model predicts…"), never as findings; the E2 measurement, once
done, converts the table's x-axis from hypothetical to located.
Deliverable: table + ~150 words of Discussion framing. Effort: 1 day.
E7 (the networked experiment) is then cited as the designed test of
rows 1–4 — the table is what makes the Tier-3 agenda look planned
rather than deferred.

## 5. Venue re-confirmation (checked 2026-08-31)

**The recommendation is Science Advances primary, now without
contingency (E2 parked → the paper stays simulation-only, anchored by
the docking chain); arXiv (cs.CY + cs.MA, cross-list physics.soc-ph)
at submission regardless.** NMI reverts to the scope-stretch third
option; the E2 side project itself is a natural NMI/AI-venue short
paper later, and the two cross-cite. Points re-verified:

- The Science Advances precedent is real and on point: *Emergent
  social conventions and collective bias in LLM populations*, Sci.
  Adv. 11, eadu9368 (2025) — pure-simulation, AI-population social
  dynamics, same genre shelf the paper argues from.
- The topic has, if anything, risen in salience since the brief was
  written: assistant sycophancy is now a mainstream research and
  policy concern (dedicated benchmarks, dependence studies, prominent
  coverage), which strengthens the machine-behaviour cover-letter
  framing at both SciAdv and NMI — and shortens the novelty window
  for E2 (§1).
- No new venue changes the calculus. The excluded list (PNAS/Nexus,
  JASSS/JCSS, young non-indexed venues) stays excluded; Nature
  Communications remains the reach-plus-indexing backup with the
  known APC/speed costs.
- Decision rule (updated 2026-08-31, E2 parked): submit Science
  Advances; no NMI presubmission inquiry is needed for this paper.
  The inquiry idea transfers to the E2 side project, for which
  measured sycophancy of deployed systems *is* squarely the genre.

## 6. Summary of new artifacts this plan creates

| Step | Files |
|---|---|
| E1 | `experiments/docking_fit/{targets.json, fit_docking.py}`, fitted-parameter sweep results dir, SI section |
| E2-lite | Discussion paragraph + 3–5 References.bib entries |
| E3 | Table S7 revision in `PNAS_Paper/SI_Appendix.tex`, References.bib entries |
| E4 | Predictions table + framing text (Discussion; drafted in the redraft pack) |
| E2 (side project, own track) | `experiments/alpha_elicitation/*` per protocol §9, OSF prereg, its own paper — nothing in this paper except the agenda sentence / optional companion citation |

Each lands as its own PR; E2's stimuli/code freeze precedes its data
collection (prereg discipline); none of them may alter any frozen
result in `results-mechfix/` or its companions.
