# DisasterAI Codebase Review — Findings & Review Protocol

**Review date:** 2026-07-06
**Reviewed branch:** `main` at `daead2b` (latest commit in the repository; merge of PR #50,
2026-04-15). All `claude/*` feature branches are older than `main` and appear merged or
superseded. This review therefore audits `main`.

**Scope:** `DisasterAI_Model.py` (model, agents, metrics, experiments),
`test_filter_bubbles.py` (Goldilocks pipeline), `simulate.py` (CI runner),
`plot_results.py`, `METHODS_PAPER.md`, CI workflows.

**Method:** full read of the agent/model/metric code paths, cross-check against
`METHODS_PAPER.md`, plus instrumented smoke runs (30–60 ticks, 40 agents, α ∈ {0, 0.5, 1})
to verify that the mechanisms execute and to test suspected defects empirically.
No code was changed; this document is the deliverable.

---

## 1. Verdict summary

The simulation **runs end-to-end without crashes**: beliefs initialize, agents sense,
query, update via the D/δ acceptance + Bayesian blend, send relief, and receive delayed
rewards; SECI/AECI/MAE/unmet-needs time series populate; the α parameter reaches the AI
report formula exactly as documented (`r = (1−α)·t + α·b`).

However, the review found **one mechanism-level defect that materially distorts the
Q-learning results** (Finding A1), **two conceptual shortcuts that should be disclosed or
redesigned** (A2, A3), **five implementation bugs** (B1–B5), and **substantial
metric-naming/sign inconsistencies between code and paper** (C1–C10) that must be resolved
before results are publishable. Several documented model features (shocks, baseline drift,
agent movement) do not exist in the code.

Priority order for remediation: **A1 → B1 → C1/C2 → B3 → B4 → everything else.**

---

## 2. Mechanism assessment (Q-learning & alignment loop)

### What demonstrably works

- **Three-mode ε-greedy source selection** (`self_action` / `human` / `ai`) with standard
  Q-updates `Q ← Q + lr·(r − Q)` from three feedback channels: delayed relief outcome
  (15–25 ticks, `process_reward`), fast information-quality cross-referencing
  (`evaluate_pending_info` / `evaluate_information_quality`), and multi-source
  triangulation.
- **Type asymmetry is wired correctly in the reward definition**: explorers are rewarded
  on *accuracy* (report vs. reference/truth), exploiters on *confirmation* (report vs.
  prior / network consensus) — `DisasterAI_Model.py:573-584` and `866-869`.
- **Directional sanity check passes for exploiters**: in seeded 60-tick runs, exploiter
  Q(ai) rises monotonically with α (0.63 → 0.98 → 1.16 for α = 0/0.5/1), i.e. confirming
  AI is increasingly preferred by confirmation-seeking agents, and exploiter SECI is
  negative (social echo chamber forms). The α interpolation, D/δ acceptance, and
  precision-weighted belief blend all execute as designed.

### A1 (CRITICAL): `triangulate_sources` injects unbounded additive Q-drift that dominates learning

> **STATUS: FIXED on this branch.** The mode-Q bump was removed; the trust nudge is now
> one-shot (each report triangulates at most once) and explorer-only. Validation (same
> seeds as below): all Q-values within [−0.36, 0.69]; toggling triangulation no longer
> changes the Q(ai)/Q(human) ordering; exploiter Q(ai) still rises with α (0.16 → 0.37)
> and explorer Q(ai) now *falls* at α = 1 (0.54 → 0.31) — the intended asymmetry, which
> the drift had been masking. Paper-scale re-validation (Stage 4) still required.

`DisasterAI_Model.py:939-987`. Each tick, every pair of pending reports on the same cell
within a 5-tick window adds ±0.05 **additively** to the mode Q-value (`Q ← clip(Q + 0.05,
−2, 2)`), with no contraction toward a reward target. Because pending items persist up to
15–30 ticks, the *same* pair of reports is re-rewarded on every tick of the overlap
window, and exploiter friend networks (which agree by construction) generate consensus
continuously.

**Empirical verification** (50 ticks, 40 agents, α = 0.5, seed 2):

| condition | Q(human) | Q(ai) | Q(self) |
|---|---|---|---|
| with triangulation | 1.103 | 0.856 | 0.228 |
| triangulation disabled | 0.332 | 0.456 | 0.196 |

The drift roughly triples Q(human), pushes Q-values above the maximum achievable
reward-based target (≈ 1.0), and **reverses the mode ordering** (Q(ai) > Q(human) without
drift; Q(human) > Q(ai) with it). Source selection — the core mechanism the paper's
echo-chamber story depends on — is currently driven more by this drift than by the
designed accuracy/confirmation rewards. Note the drift also *rewards echo-chamber
consensus itself* (agreeing friends bump each other's Q every tick), which conflates
cause and measurement.

**Suggested fix (choose one):** (a) evaluate each pending item in triangulation at most
once (mark items, as the other evaluators do); (b) convert to a proper contraction update
`Q ← Q + lr·(±r_tri − Q)`; (c) drop the ±0.05 mode-Q bump and keep only the small trust
nudge. Re-run the α sweep afterwards — headline results will likely change.

### A2 (HIGH, disclose or redesign): explorers score sources against a ground-truth oracle

> **STATUS: REDESIGNED on this branch.** The instant/perfect ground-truth reference was
> replaced with an exogenous "situation report" verification channel: for explorers'
> accepted remote reports, verification arrives with per-tick probability
> `verification_probability` (new model parameter, default 0.3) after the 3-tick minimum
> lag, carries the same noise model as direct sensing (±1 w.p. 0.2), and is evaluated
> against the *current* disaster state. Unverified items expire at 30 ticks with no
> learning signal — explicitly no fallback to prior-comparison. Documented in
> METHODS_PAPER.md/.tex. Validation (3 seeds × 100 ticks): outcomes comparable to the
> old oracle at the default (explorer Q(ai) 0.40/0.43 vs 0.38/0.40 at α=0/1; MAE and
> SECI in the same range); the p_verify sensitivity hook is live (p=0/0.3/1.0 →
> explorer Q(ai) −0.05/0.51/0.39 at α=1). Run the p_verify ∈ {0.15, 0.3, 0.6}
> sensitivity sweep at paper scale for the supplement (Stage 4).

`DisasterAI_Model.py:814-816`: for accepted remote reports, explorers' accuracy score is
computed directly against `model.disaster_grid` — the true state — 3 ticks after receipt.
The surrounding comments justify a 30-tick window so explorers can "move and sense the
cell", but **agents never move** (see A3), so this is not an approximation of physical
verification; it is oracle access. It is also the main driver of the intended
"explorers punish confirming AI" asymmetry. This is a defensible modeling choice
(explorers as agents with independent verification capacity) but it is currently
undocumented in METHODS_PAPER.md and makes the triangulation channel largely redundant
for explorers. Either document it explicitly as an assumption, or replace it with noisy/
probabilistic verification and confirm the sweet-spot result survives.

### A3 (HIGH, disclose): agents are immobile and sense only their own cell

There is no movement code anywhere (`grid.move_agent` never called; `pos` set once at
init). `sensing radius = 0` (`DisasterAI_Model.py:135, 407, 1062`), so each agent has
direct ground truth for exactly one fixed cell for the entire run. Consequences:

- The comment in `test_filter_bubbles.py:330-332` ("agents move every tick so their end
  position is arbitrary") is false; `initial_pos == pos` always.
- "Exploration" is purely informational (query targeting), not spatial. Fine, but the
  paper should not imply movement.
- All spatial-periphery results are about *fixed* spawn positions — consistent with the
  stated "structural periphery" framing, but only accidentally.

### A4 (MEDIUM): explorer AI-trust is nearly insensitive to α

In seeded 60-tick runs, mean explorer trust in AI was 0.86 / 0.85 / 0.82 at α = 0 / 0.5 /
1, and explorer Q(ai) was ~flat (0.51 / 0.62 / 0.58). The strong trust-decay pull toward
0.5 each tick (`apply_trust_decay`, rate 0.012) plus symmetric-ish fast rewards may be
washing out the α signal for explorers. Also note the same mini-sweep showed exploiter
SECI *less* negative at α = 1 than α = 0 — opposite to H1 — though 60 ticks / 40 agents
is far below paper scale. **Verify both at paper scale** (200 ticks, N=100, 20 reps)
before trusting the hypothesis-level conclusions.

---

## 3. Implementation bugs

### B1 (CRITICAL): shock parameters are dead — Experiment C's shock dimension is a no-op, and the disaster is nearly static

> **STATUS: FIXED on this branch.** `update_disaster` now implements the documented
> mechanics: per-cell stochastic drift toward `baseline_grid` (p = 0.05·pace per tick)
> plus random patch shocks (Moore radius 2, centre ±`shock_magnitude`, distance-
> attenuated) firing with p = `shock_probability`·pace, where pace = 0.5/1.0/2.0 for
> `disaster_dynamics` = 1/2/3 (0 remains static). Defaults (p=0.1, mag=2, dd=2) match
> METHODS line 7. Validation (10 seeds, 100 ticks, 20×20, environment isolated):
> cumulative level-changes mean 43/105/182 for dd=1/2/3 (monotonic; was ~5–9 single-cell
> bumps); `shock_magnitude` 1/2/3 now yields 4/27/174 changes and visibly different MAE —
> Experiment C's shock dimension is live. Note: end-vs-start cell diffs understate churn
> because drift *restores* shocked cells toward baseline by design; use cumulative
> changes or `event_ticks` to assess dynamism.

- `shock_probability` / `shock_magnitude` are stored (`DisasterAI_Model.py:2266-2267`)
  and swept by `experiment_disaster_dynamics` (`:4412-4420`), but `update_disaster`
  (`:2838-2885`) never reads them. All shock-magnitude conditions are statistically
  identical.
- What `update_disaster` actually does: with probability 0.05/0.10/0.20 per tick, bump
  **one random cell** upward by 1–4. There is **no decay back toward baseline** and no
  downward change ever — the disaster only grows, by at most one cell per tick.
  Empirically: 5–9 of 400 cells changed over 60 ticks. The environment is effectively
  static, which undercuts the "dynamic disaster environment forcing ongoing
  information-seeking" premise (METHODS_PAPER.md line 7 claims per-tick drift toward
  baseline plus random shocks with p=0.10, ±2 — neither exists).
- `baseline_grid` is computed and never used again.

**Fix:** implement drift-toward-baseline + shocks as documented (using the two
parameters), or rewrite the paper's environment description and drop the shock sweep.

### B2 (HIGH): tipping-point detection uses an inverted sign convention for AECI
> **STATUS: FIXED on this branch.** The `tp_*` tracking block (init + detection) was
> deleted: its sign convention was inverted for the current AECI formula and its outputs
> were never consumed. Transition timing is measured post-hoc in test_filter_bubbles.py
> (`_first_sustained_break` / `_first_sustained_cross`) instead.

`DisasterAI_Model.py:3355-3371` marks "AI echo chamber formation" when `avg_aeci < −0.3`.
But the current AECI formula (`:3245-3334`) outputs **positive** values for echo chambers
(`heavy_err > light_err → AECI > 0`, comment at `:3259`). The detection logic was written
for the old variance-based AECI (negative = echo) and never updated. The `tp_*`
attributes are currently written but never consumed downstream, so no published figure is
affected yet — but this is a landmine for anyone who uses them. Fix the sign or delete
the block.

### B3 (HIGH): stored-prior branch is dead in `evaluate_information_quality`

> **STATUS: FIXED on this branch** (`==` → `>=`). Verified with a targeted test: a
> 7-tuple pending item whose current belief equals the reported level (fully
> contaminated) but whose stored prior strongly disagrees now yields a *negative*
> confirmation reward (Q[human] 0.0 → −0.055), i.e. the score is computed against the
> uncontaminated stored prior as intended.

`DisasterAI_Model.py:526`: `if len(item) == 6:` — but every pending item is a 7-tuple
(verified empirically: all 2,227 pending items in a smoke run had length 7, since
`update_belief_bayesian` appends 7-tuples at `:1253` and `:1311`). The branch that uses
the uncontaminated stored prior for the confirmation score therefore never fires, and the
score falls back to the **current belief — already shifted toward the report being
evaluated**. This re-introduces exactly the circular-evaluation bias the stored prior was
added to fix (its sibling `evaluate_pending_info` correctly uses `len(item) >= 6` at
`:711`). One-character fix: `==` → `>=`. Affects exploiter confirmation rewards whenever
an item for the agent's own cell is evaluated during sensing.

### B4 (HIGH): x/y swapped in near/far spatial-coverage metrics
> **STATUS: FIXED on this branch.** The mgrid unpacking now matches the [x, y] array
> layout. Verified with the acceptance test: for epicenter (25, 5) on a 30×30 grid the
> near mask now centres on the true epicenter (centroid distance 3.4 cells, vs 24.9 to
> the diagonal reflection); the old code centred it on the reflection. NOTE: spatial
> near/far scalars are computed at simulation time, so any sweep JSONs produced before
> this fix carry the buggy values — regenerate spatial-coverage/periphery figures from
> a post-fix run.

`test_filter_bubbles.py:310-317`: the arrays are indexed `[x, y]` (tokens fill
`tok[pos[0], pos[1]]`, disaster grid is `[x, y]`), but the distance field is built as
`_Yg, _Xg = np.mgrid[0:_H, 0:_W]` and then `dist = sqrt((_Xg−ex)² + (_Yg−ey)²)` — i.e.
for array cell `[i, j]` it computes `sqrt((j−ex)² + (i−ey)²)`: coordinates transposed.
Whenever the epicenter is off-diagonal (`ex ≠ ey`), the near/far masks are reflected
across the grid diagonal, corrupting `near_cell_deficit`, `far_cell_deficit`,
`near_cell_aid`, `far_cell_aid` — the inputs to the spatial-equity figures. The
*agent*-distance computation eight lines later (`:333-334`) is correct, so the two
periphery analyses in the same function disagree. Fix the mgrid ordering (swap `_Xg`/
`_Yg` or index consistently) and regenerate all spatial-coverage figures.

### B5 (MEDIUM): `max_aeci_variance` is never populated
> **STATUS: FIXED on this branch** (`ndim == 3` → `ndim == 2 and shape[1] > 1`).
> Verified: a 2-run aggregation now returns a populated `max_aeci_variance` list.

`DisasterAI_Model.py:4210-4224`: the guard is `aeci_variance_data.ndim == 3 and
shape[2] > 1`, but each run's array is 2-D `(ticks, 2)` (built by `np.column_stack` at
`:4116`). The branch never executes; every run prints "Unexpected AECI variance data
shape", `max_aeci_variance_per_run` stays empty, and
`plot_max_aeci_variance_by_alignment` has nothing to plot. Fix: `ndim == 2 and
shape[1] > 1`.

### B6 (LOW, latent): non-square grids would break silently
> **STATUS: FIXED on this branch** in the two model locations (`initialize_beliefs`,
> `AIAgent.sense_environment`); the test_filter_bubbles `_H/_W` instance was fixed as
> part of B4. Verified: a 25×20 grid run initialises all 500 cell beliefs and completes.
> Plot-only helpers were not audited for non-square grids.

`disaster_grid = np.zeros((width, height))`, but two places unpack
`height, width = disaster_grid.shape` (swapped): `HumanAgent.initialize_beliefs`
(`:115`) and `AIAgent.sense_environment` (`:2015`); same pattern with `_H, _W` in
`test_filter_bubbles.py:310`. Harmless at 30×30; wrong for any `width ≠ height`. Either
fix the unpacking or assert squareness in `DisasterModel.__init__`.

### B7 (MEDIUM): `simulate.py` (the CI runner) is unseeded
> **STATUS: FIXED on this branch.** `simulate.py` now seeds each replicate with
> `--seed_base + i` (default 0), giving reproducible runs and common random numbers
> across conditions. Verified: two runs with identical seeds produce byte-identical
> output.

`simulate.run_one`/`main` never set `random.seed`/`np.random.seed` — unlike
`test_filter_bubbles.run_replicated` (seeds each replicate with its index, `:517-524`)
and `simulation_generator` (`:4101-4102`). CI results from `simulate.py` are
non-reproducible, and α conditions don't share common random numbers, inflating
between-condition variance. METHODS_PAPER.md line 58 ("each using a different random
seed") over-claims for this path. Fix: add a `--seed-base` argument and seed each
replicate as `seed_base + i`.

---

## 4. Metric consistency & rationale (SECI, AECI, MAE, precision, unmet needs)

### C1 (CRITICAL for the paper): three different metrics share the name "AECI", with conflicting sign conventions

| Where | Formula | Echo-chamber sign |
|---|---|---|
| METHODS_PAPER.md:41-45 ("AECI") | accepted_AI / (accepted_AI + accepted_human) | high = reliance (0..1 scale) |
| `model.aeci_data` (`:3245-3334`) — plotted as "AECI" in all time-series panels | median split by `accum_calls_ai`; difference of confidence-weighted belief error, AI-heavy vs AI-light | **positive** = echo chamber |
| `model.aeci_variance_data` (`calculate_aeci_variance`, `:2504-2577`) — "AECI-Var", used in the Goldilocks composite (`test_filter_bubbles.py:581-586`) | belief variance of AI-reliant agents vs global | **negative** = echo chamber |

The paper's formula corresponds to what the code calls `retain_aeci` — which is *not*
what any headline figure plots. SECI and AECI-Var use −1 = echo; time-series AECI uses
+1 = echo; both are plotted on adjacent panels labeled "(-1 to +1)". A reader (or a
co-author) cannot currently tell which construct any figure shows.

**Fix:** pick one name per construct — e.g. `AECI-Acc` (acceptance share, the paper
formula = `retain_aeci`), `AECI-Err` (confidence-weighted error split, current
`aeci_data`), `AECI-Var` (variance ratio) — align all sign conventions so negative =
echo chamber everywhere (flip `aeci_data`'s sign or its interpretation), update every
plot label, and state in the paper exactly which enters the Goldilocks composite.

### C2 (CRITICAL for the paper): the SECI formula in METHODS_PAPER.md contradicts both its own text and the code

Paper (line 37): `SECI = 1 − Var(friends)/Var(all)` → echo chamber (low friend variance)
gives **positive** SECI, yet the very next sentence says "SECI < 0 indicates friends hold
more homogeneous beliefs". The code (`:3014-3056`) computes
`(Var_community − Var_global)/Var_global` for the negative half (echo → negative, matching
the text, i.e. the *negation* of the paper's formula) with an asymmetric positive half
normalized by `(5 − Var_global)`. The code also differs from the paper in three ways that
should be documented: (i) it pools **network components** (type-homogeneous communities),
not ego-network friends; (ii) it filters to beliefs with level ≥ 1; (iii) global variance
is pooled across both types while community variances are type-averaged. The code's
choice is defensible (Cinelli et al. 2021 framing, cited in the code) — the paper's
formula block just needs to be rewritten to match it.

### C3 (HIGH): two MAE definitions coexist

- `model.belief_error_data` (`:3181-3222`): MAE over **all 900 cells**.
- `simulate.py:104-121` / `test_filter_bubbles.py:262-276`: MAE over **disaster cells
  (true level ≥ 1) only**, with a good in-code rationale (confidence on L0 cells dominated
  the all-cell average at high α).

The paper (line 47, "averaged over cells with non-zero severity") matches the pipeline
definition, but any figure produced from `belief_error_data` (e.g.
`plot_belief_error_evolution`) uses the other one. State in the paper which definition
each figure uses, or make the model metric match the pipeline.

### C4 (MEDIUM): two relief-precision definitions coexist

- `process_reward` (`:1717-1725`): token correct iff true level ≥ 3 **15–25 ticks after
  targeting** (feeds `correct_targets`/`incorrect_targets` → assist stats and Q rewards).
- `test_filter_bubbles.py:240-249` / `simulate.py:80-89`: token correct iff true level ≥ 3
  **at placement tick** (feeds `prec_*`).

With the near-static disaster (B1) these nearly coincide today, but they will diverge the
moment B1 is fixed. Choose one for the paper and name the other differently.

### C5 (MEDIUM): AECI-Var's "AI-reliant" classification contradicts the code's own comments, and its signal is tiny

`calculate_aeci_variance` classifies agents by `accepted_ai ≥ 3` (`:2513-2519`), a
counter **reset every 5 ticks** (`:3489-3494`); the agent-init comment (`:103`) says
`cum_accepted_ai` "used for AECI classification", and the `aeci_data` split uses yet
another basis (`accum_calls_ai`, `:3303`). Three classification bases across the AECI
family. Additionally, empirical AECI-Var magnitudes are ~0.001–0.015 versus |SECI| ~
0.2–0.3; after the range normalization in `compute_goldilocks_metrics`
(`test_filter_bubbles.py:604-630`), this near-noise component contributes equally to the
composite that determines α*. **Recommend:** unify the classification basis, and run a
sensitivity check of α* with (a) AECI-Var excluded, (b) `retain_aeci` substituted — if α*
moves materially, the composite is not robust.

### C6 (LOW): unmet-needs threshold documented as L4+, implemented as L3+

`test_filter_bubbles.py:29` says "high-need cells (≥L4)"; `DisasterAI_Model.py:2943` uses
`>= 3`. Also note the definition counts a cell as unmet **per tick** unless it receives a
token that same tick — with 100 agents × up to 5 tokens each per tick, this is generous;
fine, but say so in the paper.

### C11 (construct validity — added post-Stage-2 sanity check)

Empirical check (9 toy runs: α ∈ {0, 0.5, 1} × 3 seeds, 80 ticks, 40 agents, all
Stage-1–2 fixes active), correlations of per-run steady-state values:

| pair | r |
|---|---|
| SECI vs AECI-Var | +0.37 |
| SECI vs AECI-Err | −0.57 |
| **AECI-Var vs AECI-Err** | **−0.72** |

**The indices do not measure one phenomenon.** SECI measures *social fragmentation*
(within-community homogeneity ⟺ between-community disagreement). AECI-Var measures
*population-level homogenization among AI users* (monoculture risk). AECI-Err measures
*epistemic harm* (confidently-wrong beliefs among AI-influenced agents). AECI-Acc is an
exposure measure. The two AI indices are strongly ANTI-correlated: where AI reliance
homogenizes beliefs (AECI-Var < 0), AI-heavy agents tend to be relatively MORE accurate
(AECI-Err > 0). Treating them as interchangeable "AECI" readings — or summing either
with SECI without stating which construct is meant — invites misinterpretation. The
composite |SECI| + |AECI-Var| is defensible only when framed as the sum of two distinct
harms (fragmentation + monoculture), not as "total echo chamber".

**Truth-convergence blind spot.** Variance-based indices (SECI, AECI-Var) register
convergence on the TRUE state as an echo chamber: a perfectly truthful AI that teaches
the whole population the truth maximizes |AECI-Var|. The echo-chamber literature requires
*insulation from correction*, not mere homogeneity. AECI-Err is the construct that
operationalizes this (homogeneous AND wrong); headline echo-chamber claims should be
conditioned on error (the total_score composite partially does this via MAE).

**"AI-reliant" definition audit.** Current: median split by `cum_accepted_ai`
(cumulative D/δ-accepted AI belief updates). In the toy runs the label is meaningful in
absolute terms (top half ≈ 57–68 % of queries to AI; median ≈ 550–700 accepted updates
in 80 ticks) — report these per α in the paper. Two residual issues:

1. **Type-composition confound in AECI-Var**: the split pools both agent types, and the
   top half's exploiter share swings from ~25 % (α = 0) to 45–75 % (α = 1). Across the α
   sweep, AECI-Var therefore partly measures *which type self-selects into AI use*, not
   what AI does to beliefs. Recommended fix: split within type (as AECI-Err does) and
   average — requires a re-run to take effect.
2. **Endogeneity**: acceptance counts are outcomes of the trust/Q dynamics, so
   "AI-reliant" is self-selected. Fine for descriptive indices; causal phrasing
   ("AI causes homogenization") should be avoided or hedged.

### C12 (mechanism diagnosis): why explorers keep relying on AI even when it is wrong

> **STATUS: counterfactual IMPLEMENTED on this branch.** New `salience_weight`
> model parameter (default 0.0 = uniform baseline; CLI: `--salience-weight` in
> test_filter_bubbles.py, `--salience_weight` in simulate.py) scales the verified-
> evaluation learning rate by (1−s) + s·(max(truth, reported)+1)/6. Validation
> (3 seeds × 100 ticks): the explorer α-gradient FLIPS SIGN — uniform evaluation
> (s=0): Q(ai) 0.34 → 0.42 as α goes 0 → 1 (confirming AI mildly preferred);
> full salience (s=1): Q(ai) 0.33 → 0.27 and AI-trust 0.83 → 0.77 (confirming AI
> punished). Documented in METHODS_PAPER.md/.tex and SUPPLEMENTARY.md. For the
> paper: run the α sweep at s ∈ {0, 1} and present trust-persistence-vs-collapse
> as the C12 figure.

Observed in the paper-scale run (28817810131): explorer AI query share is flat at
~0.55–0.60 across ALL α (always above 50 %), explorer AI-trust leads friend-trust from
t = 0 in 100 % of runs, yet explorer SECI jumps to ≈ 0.29 at α ≥ 0.9.

**Diagnosis (instrumented probe, 80 ticks, 40 agents, α ∈ {0, 1}): the verification
reward is base-rate dominated.** Logging every AI report delivered to explorer queries:

| | α = 0 (truthful) | α = 1 (confirming) |
|---|---|---|
| share of reported cells with true level 0 | 89 % | 91 % |
| exact-correct on those L0 cells | 97 % | 99 % |
| within ±1 on true L3+ cells | 38 % | **2 %** |
| **overall within ±1 (positive-reward zone)** | **96 %** | **93 %** |

Explorers query high-uncertainty areas, but ~90 % of those cells are truly empty, and a
confirming AI echoes the explorer's (mostly correct) "nothing there" prior — so it
passes situation-report verification on ~93 % of cells even at α = 1. Its failures
concentrate entirely on the rare high-severity cells (within ±1 accuracy collapses from
38 % → 2 % on L3+), but those few negative rewards are swamped by the empty-cell
positives. The Q-learning works exactly as designed; the reward signal itself cannot
distinguish a truthful AI from a confirming one because **accuracy per cell is not
weighted by importance**.

This is arguably the paper's most interesting mechanism finding: *a confirming AI
retains user trust because it is right about the unimportant majority; its errors
concentrate precisely on the high-severity cells that drive response performance* —
which is why explorer reliance stays flat while unmet needs explode at α ≥ 0.9.

**Options:** (a) keep as-is and make it a headline finding (defensible; mirrors
real-world AI assistants that stay trusted by being right on easy queries); (b) add
severity-weighted (salience) verification — scale the accuracy reward by
max(truth, reported)/5 so being wrong about a disaster is more memorable than being
right about nothing (also behaviorally defensible via negativity/salience bias). Option
(b) would likely restore explorer α-sensitivity and is worth a parameterized variant;
it changes results and needs a re-run.

### C7–C10 (paper/code rationale gaps to document)

- **C7.** METHODS line 13 says only exploratory agents get fast information-quality
  feedback; in code both types do, with different reward bases (accuracy vs.
  confirmation). The code's version is arguably the better design — update the paper.
- **C8.** METHODS line 21 claims α "isolates the confirmation effect from other AI
  failure modes (… bias)", but `AIAgent.sense_environment` sets sensing noise
  `noise_prob = 0.1·α` (`:2029-2032`) — high-α AI is simultaneously more confirming *and*
  noisier: a confound in the sweep. Recommend removing the α-dependence of noise (fixed
  0.1 or 0) so α is a pure confirmation dial.
- **C9.** AI coverage/guessing: each AI senses 15 % of the grid but answers queries about
  unsensed cells with fabricated estimates 75 % of the time (`:2107-2154`, spatial
  interpolation or position-based random draws). Meanwhile the human evaluation code
  assumes "AI has broad knowledge → source_knowledge_conf = 1.0" (`:557-559`, `:853-854`),
  so agents update trust/Q at full strength against reports that are often guesses — and
  at α = 0 the "truthful" AI still emits random-ish values for most of the grid. Either
  lower the guess rate, have the AI report only sensed cells, or document this as a
  deliberate "AI overconfidence" feature.
- **C10.** For exploitative callers the AI confirms the **network-consensus** belief, not
  the individual prior (`:2169-2177`), i.e. the AI is assumed to know the caller's
  friends' beliefs (a social-media-platform analogy). This is a strong, load-bearing
  assumption (the code comments say H1 fails without it) and is absent from
  METHODS_PAPER.md. It must be stated and justified in the paper. Likewise the trust-
  widened acceptance `D_eff` applies only to *friends* in code (`:1242-1248`), not "social
  contacts" as the paper says.

---

## 5. Dead code & vestigial parameters (omissions/cruft)

| Item | Location | Note |
|---|---|---|
| `decay_trust` | `:1949-1953` | Never called; would raise `KeyError: 'ai'` if ever called (`self.trust["ai"]` doesn't exist — trust keys are `A_k`). Delete. |
| `smooth_friend_trust` | `:1935-1946` | Never called. Delete or wire in. |
| `find_exploration_targets` / `exploration_targets` | `:1093-1190` | Never called from the step flow. Delete. |
| `share_confirming` | param everywhere, `simulate.py` sets 0.7 | Stored on every agent, **never used in any logic**. Misleading — remove or implement. |
| `lambda_parameter`, `low_trust_amplification_factor`, `exploitative_correction_factor`, `trust_update_mode` | `:2231-2275` | Stored, never used. Remove. |
| `exploit_friend_bias`, `exploit_self_bias` | `:2247-2248` | Accepted, never used (biases explicitly removed at `:1473-1475`). Remove. |
| `tp_*` tipping attributes | `:2319-2334`, `:3355-3371` | Written, never read (and sign-buggy, B2). Fix or delete. |
| `baseline_grid` | `:2338, 2349-2351` | Computed, never used after init (see B1). |
| `AIAgent.memory` | `:2038` | Grows unboundedly (keyed by `(tick, cell)`, never pruned). Harmless at current scale; prune keys older than 1 tick. |
| Duplicate `self.q_table = {}` | `:60` and `:72` | Cosmetic. |
| `aggregate_simulation_results` run counter | `:4183-4189` | Increments `successful_runs` before checking for the empty (error) result — failed runs are counted as successful in logs. Cosmetic. |
| Stale comment "AECI (AI Call Ratio)" | `:5979`, `:4359` | aeci_data has not been a call ratio for several revisions. |

---

## 6. Review protocol (ordered, with acceptance criteria)

Work through the stages in order; each stage gates the next. Estimated effort is for one
person familiar with the code.

### Stage 0 — Freeze a baseline (½ day)
1. Tag current `main` (`git tag pre-review-baseline`).
2. Run the seeded mini-sweep used in this review (α ∈ {0, 0.5, 1}, 60 ticks, 40 agents,
   fixed seed) and archive the metric outputs. These are your "before" numbers for
   detecting result shifts from each fix.

### Stage 1 — Mechanism fixes (1–2 days)
3. **Fix A1** (triangulation drift): make triangulation a one-shot, contraction-style
   update or remove the mode-Q bump.
   *Acceptance:* with all sources truthful, mode Q-values remain within the reward range
   [−1, 1]; disabling triangulation no longer changes the Q(ai)/Q(human) *ordering* at
   α = 0.5 by more than noise.
   **✅ DONE on this branch — both acceptance criteria verified (see A1 status note).**
4. **Fix B3** (`==` → `>=` at `:526`).
   *Acceptance:* add a unit test that appends a 7-tuple pending item and asserts the
   confirmation score is computed against the stored prior, not the current belief.
   **✅ DONE on this branch — targeted test passes (see B3 status note).**
5. **Decide A2** (explorer ground-truth oracle): document as an explicit assumption in
   METHODS, or replace with noisy verification.
   *Acceptance:* a sentence in METHODS, or a re-run showing the sweet spot survives noisy
   verification.
   **✅ DONE on this branch — replaced with the situation-report channel and documented
   in METHODS (see A2 status note); paper-scale sweet-spot confirmation still pending
   in Stage 4.**
6. **Fix B1** (disaster dynamics): implement drift-toward-baseline + shocks using
   `shock_probability`/`shock_magnitude`, or rewrite METHODS line 7 and delete the shock
   sweep from Experiment C.
   *Acceptance:* over 100 ticks ≥ 5 % of cells change at dynamics = 2 (or the paper no
   longer claims a dynamic environment); Experiment C conditions differ statistically.
   **✅ DONE on this branch — drift + patch shocks implemented; churn is monotonic in
   both `disaster_dynamics` and `shock_magnitude` (see B1 status note; cumulative churn
   ≈ 26 % of cell-levels per 100 ticks at dd = 2, though end-vs-start diffs are lower
   because drift restores toward baseline by design).**

### Stage 2 — Metric unification (1 day)
7. **Resolve C1**: one name per AECI construct; one sign convention (recommend negative =
   echo chamber for all three); relabel every plot; state in METHODS which construct is in
   the Goldilocks composite.
   **✅ DONE on this branch.** Names: AECI-Acc (acceptance share, `retain_aeci`),
   AECI-Err (confidence-weighted error split, `aeci_data`, sign FLIPPED so negative =
   echo chamber), AECI-Var (variance ratio, composite component). All plot labels
   updated across DisasterAI_Model.py, test_filter_bubbles.py, plot_results.py.
   JSONs now carry a `conventions.aeci_err_sign` marker; files written before the
   flip are auto-converted on load, so run 28808662941's artifacts stay usable.
   Bonus fix: `plot_aeci_evolution` plotted AECI-Err under query-ratio labels with
   ylim(0,1) clipping all negative values — it now plots the actual
   `ai_query_ratio_*` series.
8. **Resolve C2**: rewrite the SECI formula block in METHODS to match the code
   (component-based, L1+ filter, asymmetric normalization).
   **✅ DONE on this branch** — METHODS_PAPER.md, SUPPLEMENTARY.md, and
   SUPPLEMENTARY.tex now carry the code-matching piecewise formula
   (METHODS_PAPER.tex already had it).
9. **Resolve C5**: single classification basis for "AI-reliant" (recommend
   `cum_accepted_ai`, which is what the init comment promises); re-run and report the α*
   sensitivity check (composite with/without AECI-Var, and with retain_aeci substituted).
   **✅ DONE on this branch.** Both AECI-Err and AECI-Var now median-split by
   `cum_accepted_ai` (AECI-Var previously used the per-period `accepted_ai >= 3` rule,
   AECI-Err used `accum_calls_ai`). `alpha_star_sensitivity()` reports α* under six
   composite variants ({SECI+AECI-Var, SECI only, SECI+AECI-Err} × {±MAE}), printed in
   the collect summary and plotted as alpha_star_sensitivity.png. NOTE: the
   classification change shifts AECI-Var/AECI-Err values — the paper-scale sweep must
   be re-run (or replotted via the Replot workflow for α*-sensitivity only, since
   |AECI| composites are sign-invariant but NOT classification-invariant: full re-run
   required for final numbers).
10. **Resolve C3/C4/C6**: pick one MAE and one precision definition for the paper; fix the
    L3/L4 doc string.
    *Acceptance for stage:* a table in SUPPLEMENTARY listing every reported metric, its
    exact formula, its code location, and its sign convention — with no duplicates.
    **✅ DONE on this branch.** Reported MAE = disaster-cell (L1+) definition; reported
    precision = placement-time definition (delayed assessment feeds rewards only);
    unmet needs documented as L3+ everywhere (docstrings fixed). SUPPLEMENTARY.md
    S5.0 now contains the full metrics reference table (stage acceptance criterion).

### Stage 3 — Secondary bug fixes (½ day)
11. Fix B4 (spatial x/y swap) and regenerate all spatial-coverage/periphery figures.
    *Acceptance:* unit test — place epicenter at (25, 5) on a 30×30 grid, assert the
    near-mask centres on the epicenter (edge clipping shifts the centroid inward, so
    test against the diagonal reflection, not a 2-cell radius).
    **✅ DONE on this branch (see B4 status note); figure regeneration needs a
    post-fix sweep run.**
12. Fix B2 (tipping sign) or delete the `tp_*` block.
    **✅ DONE on this branch — block deleted (see B2 status note).**
13. Fix B5 (`ndim == 2`), B6 (shape unpacking or squareness assert), B7 (seed
    `simulate.py`).
    **✅ DONE on this branch (see B5/B6/B7 status notes).**
14. Delete dead code / vestigial params from §5 (esp. `share_confirming`, which implies a
    mechanism that doesn't exist).
    **◐ PARTIAL on this branch: dead methods (`decay_trust`, `smooth_friend_trust`,
    `find_exploration_targets`), the `tp_*` block, and stale "AI Call Ratio" comments
    are deleted. Vestigial constructor parameters (`share_confirming`,
    `lambda_parameter`, `low_trust_amplification_factor`,
    `exploitative_correction_factor`, `trust_update_mode`, `exploit_friend_bias`,
    `exploit_self_bias`) are still accepted-but-ignored — removing them breaks every
    caller (simulate.py, test_filter_bubbles.py, notebooks), so batch that with the
    Stage-2 metric renaming, and decide whether `share_confirming` should be
    implemented instead of removed.**

### Stage 4 — Re-validation at paper scale (compute-bound)
15. Re-run the primary α sweep (200 ticks, N = 100, ≥ 20 seeded reps) via
    `test_filter_bubbles.py --primary-only`.
16. Check, at paper scale, the three behavioral questions this review could only probe at
    toy scale:
    a. Exploiter SECI more negative at high α than at α = 0 (H1 direction) — the 60-tick
       probe showed the *opposite*;
    b. Explorer trust/Q(ai) responsive to α (currently ~flat, A4);
    c. α* stable across the composite-definition sensitivity checks from step 9.
17. Confirm the gap-sweep invariant (`_gap_d_delta`) and rerun only if Stage 1 changed
    agent behavior (it will have).

### Stage 5 — Paper reconciliation (½ day)
18. Sweep METHODS_PAPER.md/tex and SUPPLEMENTARY against the fixed code line by line:
    environment dynamics (B1), feedback structure (C7), α purity (C8), AI
    coverage/guessing (C9), network-consensus confirmation and friend-only D_eff (C10),
    immobility of agents (A3), seeding claims (B7).
19. Have a co-author independently re-derive each figure's metric from the code location
    listed in the Stage-2 table.

---

## 7. Minor observations (no action strictly required)

- `update_trust_for_accuracy` (`:1955-2000`) scores AI accuracy via the agent's *blended*
  belief rather than the AI's reported value — a mild attenuation of the intended signal.
- `send_relief` assigns credit for the entire relief batch to the single mode queried
  *this* tick (`:1650`), though beliefs were shaped over many ticks — a known
  credit-assignment simplification worth one sentence in METHODS.
- `report_beliefs` (human) lets agents in L4+ cells fail to respond with p = 0.1
  (`:996`) — undocumented but plausible.
- `experiment_alignment_tipping_point` (`:4356-4379`) reads identical arrays into
  "exploit" and "explor" variables from the same key (works because columns differ, but
  the duplicated `result.get("aeci", …)` lines are confusing).
- Exploiters' initial AI trust (0.10–0.25, `:2492`) is below their AI-trust decay neutral
  point (0.35, `:1207`), so decay *raises* exploiter AI trust early on — intended?
- The α = 0 condition still has AI reporting guesses for unsensed cells (C9), so "fully
  truthful" is only truthful-in-expectation for ~15 % of cells per AI per tick.
