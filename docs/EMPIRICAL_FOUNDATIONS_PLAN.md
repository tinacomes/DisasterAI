# Tier-1 empirical foundations — verification and implementation plan

Companion to `PAPER_BRIEF.md` §*Empirical foundations*. That section
lists the upgrade path (E1–E8) and its rationale; this document (i)
re-verifies, as of **2026-08-31**, that the Tier-1 steps E1–E4 are the
right pre-submission investments, (ii) makes E1, E3, and E4 concrete
enough to execute, and (iii) records the venue re-confirmation. The
full E2 protocol and implementation plan is
`docs/ALPHA_MEASUREMENT_PROTOCOL.md`.

## 1. Verification: are E1–E4 the right steps? (checked 2026-08-31)

Verdict: **yes, all four, with one rescope (E1) and a changed
execution order.** The check was against (a) what the paper's referee
model actually needs — the brief's four-layer robustness answer covers
internal validity; Tier 1 exists to answer "your α is a cartoon" with
external anchors — and (b) the 2025–26 literature.

- **E2 (α measurement of deployed systems) — confirmed, and now the
  clear priority.** The sycophancy field has exploded (multi-turn
  stance-flip benchmarks such as SYCON-Bench; agentic and domain
  sycophancy studies; mainstream coverage of assistant sycophancy),
  and the LLM-anchoring literature shows models weight in-prompt
  numbers generically. As of the check date, **no published work
  estimates a truth–belief mixing weight against controlled in-context
  ground truth** — the stance-flip benchmarks measure *whether* a
  model flips, not *how far between truth and the user's prior* its
  numeric answer lands, which is exactly the model's estimand. That is
  a real, closable novelty window: both the strongest argument for E2
  and a scoop risk that argues for doing it first. Two design
  consequences are now folded into the protocol: an anchoring-
  attribution control (confirmation vs. generic anchoring must be
  separable) and preregistration for the timestamp.
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
E5/E6 remain confirmatory-flavoured and slow, and the docking + E2
pair dominates them per unit effort.

**Execution order (revised from the brief's cost ordering):**

    E4 (1 d) → E3 (2–4 d) → E2 (2–3 wks; the long pole — start its
    pilot early) → E1 (1–2 wks; parallelises with E2's collection window)

E4 and E3 first because they are self-contained, feed the Discussion
and SI directly, and lose nothing by preceding the others. E2 before
E1 because of the novelty window and because E1's headline ("fitted
vs. default parameters leave the population results intact") is a
robustness statement that lands the same whenever it is done.

## 2. E1 — Effect-size-targeted docking (implementation)

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

**The brief's recommendation stands: Science Advances primary, with
the E2 contingency upgrading NMI to a credible co-equal, resolved by
presubmission inquiry; arXiv (cs.CY + cs.MA, cross-list
physics.soc-ph) at submission regardless.** Points re-verified:

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
- Decision rule, unchanged but now dated: if E2 is in the package,
  send the NMI presubmission inquiry **while** assembling the SciAdv
  submission; a positive NMI response makes it an author's choice
  between AI-community reach (NMI) and interdisciplinary breadth
  (SciAdv) — both defensible; absent a positive NMI signal, submit
  SciAdv without waiting.

## 6. Summary of new artifacts this plan creates

| Step | Files |
|---|---|
| E2 | `experiments/alpha_elicitation/*` (see protocol §9), OSF prereg, SI table, money-figure overlay |
| E1 | `experiments/docking_fit/{targets.json, fit_docking.py}`, fitted-parameter sweep results dir, SI section |
| E3 | Table S7 revision in `PNAS_Paper/SI_Appendix.tex`, References.bib entries |
| E4 | Predictions table + framing text (Discussion; drafted in the redraft pack) |

Each lands as its own PR; E2's stimuli/code freeze precedes its data
collection (prereg discipline); none of them may alter any frozen
result in `results-mechfix/` or its companions.
