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

`DisasterAI_Model.py:4210-4224`: the guard is `aeci_variance_data.ndim == 3 and
shape[2] > 1`, but each run's array is 2-D `(ticks, 2)` (built by `np.column_stack` at
`:4116`). The branch never executes; every run prints "Unexpected AECI variance data
shape", `max_aeci_variance_per_run` stays empty, and
`plot_max_aeci_variance_by_alignment` has nothing to plot. Fix: `ndim == 2 and
shape[1] > 1`.

### B6 (LOW, latent): non-square grids would break silently

`disaster_grid = np.zeros((width, height))`, but two places unpack
`height, width = disaster_grid.shape` (swapped): `HumanAgent.initialize_beliefs`
(`:115`) and `AIAgent.sense_environment` (`:2015`); same pattern with `_H, _W` in
`test_filter_bubbles.py:310`. Harmless at 30×30; wrong for any `width ≠ height`. Either
fix the unpacking or assert squareness in `DisasterModel.__init__`.

### B7 (MEDIUM): `simulate.py` (the CI runner) is unseeded

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
8. **Resolve C2**: rewrite the SECI formula block in METHODS to match the code
   (component-based, L1+ filter, asymmetric normalization).
9. **Resolve C5**: single classification basis for "AI-reliant" (recommend
   `cum_accepted_ai`, which is what the init comment promises); re-run and report the α*
   sensitivity check (composite with/without AECI-Var, and with retain_aeci substituted).
10. **Resolve C3/C4/C6**: pick one MAE and one precision definition for the paper; fix the
    L3/L4 doc string.
    *Acceptance for stage:* a table in SUPPLEMENTARY listing every reported metric, its
    exact formula, its code location, and its sign convention — with no duplicates.

### Stage 3 — Secondary bug fixes (½ day)
11. Fix B4 (spatial x/y swap) and regenerate all spatial-coverage/periphery figures.
    *Acceptance:* unit test — place epicenter at (25, 5) on a 30×30 grid, assert the
    near-mask centroid is within 2 cells of the epicenter.
12. Fix B2 (tipping sign) or delete the `tp_*` block.
13. Fix B5 (`ndim == 2`), B6 (shape unpacking or squareness assert), B7 (seed
    `simulate.py`).
14. Delete dead code / vestigial params from §5 (esp. `share_confirming`, which implies a
    mechanism that doesn't exist).

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
