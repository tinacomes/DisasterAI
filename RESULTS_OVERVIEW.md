# Results Overview — Fixed Model (2026-08-20)

Consolidated overview of the current results after the model correction of
2026-08-19, for author review. All headline numbers below come from the
**fixed, type-agnostic model** (verification sweep); the legacy and ablation
runs are reported for comparison. Steady state = mean over the last 75 ticks;
N = 20 seed-paired replications per α level; 200 ticks per run.

## 1. The three runs

| Run | Directory | Code | AI confirmation target | Purpose |
|---|---|---|---|---|
| 32040117179 (2026-08-17/18) | `results/` | `0e05139` (legacy) | consensus for exploitative callers | Legacy record; basis of the NHB draft |
| **32298278561 (2026-08-19)** | **`results-verification/`** | **`55d4b2b` (fixed)** | **caller's own prior, both types** | **Canonical results for the PNAS paper** |
| 32338797843 (2026-08-20) | `results-ablation-consensus/` | `55d4b2b` (fixed) | consensus (legacy rule re-enabled) | Ablation isolating the confirmation target |

**What was fixed** (commit `55d4b2b`): (i) the AI no longer conditions its
response on the caller's cognitive type — it targets the caller's own revealed
prior for both agent types (`confirmation_target='individual'`, new default);
(ii) explorers' *rejected* remote reports are no longer accuracy-scored
against their own prior (a hidden confirmation channel) but await external
verification like accepted ones; (iii) Q/trust updates are batched per source
per tick (previously ~25 per-cell updates per query drowned the truth-grounded
relief signal 25:1); (iv) two new diagnostics: **AECI-LockIn** (belief
mobility of AI-heavy vs AI-light agents) and **L1+ belief-pool size**.

Configurations: **main model** = mobility on, spatially embedded bridged
communities, network-gated queries ("switched"); **control** = immobile,
disconnected communities, global query access ("baseline"). Sign convention:
negative SECI/AECI = echo chamber.

## 2. Finding 1 — Network-bounded access is the structural precondition

The claim survives and is now cleanly attributable **per agent type**:

- Combined SECI, main model: −0.208 (α=0) → **−0.299 / −0.312** (α=0.9/1.0).
  Control: −0.235 → −0.141 / −0.122 (monotone *shallowing*).
- Exploitative communities: under **global access** their echo chamber
  *dissolves* at high alignment (SECI_exploit ≈ **+0.02** at α≥0.9); under
  **network-gated access** it persists (**−0.27 / −0.31**). Paired per-seed
  ΔSECI_exploit (main − control) = **−0.293 / −0.338** at α=0.9/1.0, 95% CI
  excluding zero.
- Exploratory communities deepen in *both* configurations at α≥0.8
  (−0.01 → −0.33 main; +0.01 → −0.30 control), so the α-gradient of the
  combined index is explorer-driven everywhere; the configuration contrast
  at high α is exploiter-driven.

**Restated mechanism (important change from the NHB draft):** bounded access
is required for a confirming AI to *preserve* the exploitative communities'
echo chamber — under global access, out-group contact erodes it. The old
explanation ("the AI recycles the community consensus back into the
community") is retired: it described the legacy consensus-targeting code,
which the ablation shows was behaviourally inert (see §6), and the archived
per-type data never supported it.

## 3. Finding 2 — Dose–response of alignment; interior optimum only under bounded access

- Explorer MAE degrades monotonically with α: 0.55 → **1.74/1.75** (main);
  0.62 → 1.94/1.95 (control). Exploiter MAE is flat (≈1.8) at all α —
  **the accuracy cost of alignment is borne entirely by the accuracy-seekers.**
- New mechanism metric: the explorers' **L1+ belief pool collapses from ~114
  to ~27 beliefs per agent** (main; 109 → 18 control) between α=0 and α=1.
  A confirming AI echoes their (mostly level-0) priors about the uncertain
  cells they query, so they stop acquiring disaster knowledge — *belief
  starvation*. Part of the SECI deepening is this pool shrinkage (SECI
  conditions on L1+ beliefs), which is why the L1+ pool is now always
  reported alongside SECI.
- **Interior α\***: in the main model all six composite definitions give an
  interior optimum, spread **[0.3, 0.6]** (tighter than the legacy run). In
  the control, the bubble-only composites (SECI, SECI+AECI-Var) select α*=1.0
  — full confirmation minimises the *echo indices* because the exploiter
  chamber dissolves — while accuracy-including composites stay interior
  (0.3–0.7). The claim "the alignment optimum is interior only under
  network-bounded access" holds for the bubble composites and should be
  stated with that qualifier.

## 4. Finding 3 — The feedback loop: capture, lock-in, and operational buffering

- **Exploiter capture**: the exploitative agents' AI query share rises
  monotonically 0.47 → **0.70** across α (legacy: 0.34 → 0.60 — the capture
  is *stronger* in the fixed model). Their AI trust rises 0.46 → 0.50.
- **Explorer non-discrimination at high α (C12)**: explorer AI trust stays
  high (0.85 → 0.81) even at α=1 because a confirming AI passes external
  verification on the ~90% of queried cells that are truly empty; the
  salience-weighted verification variant (`salience_weight>0`) remains the
  counterfactual for this and is *not* in the mainline.
- **AECI-LockIn** (new): AI-heavy explorers' beliefs freeze relative to
  AI-light peers, deepening −0.04 → **−0.16** with α. The exploiter series is
  positive at low α (relief-correction churn) and falls toward zero at high α;
  read as an α-gradient per type. **AECI-Var stays flat (≈ −0.1) at all α**:
  a variance index cannot register an individualised (own-prior) bubble, which
  is why LockIn and the pool size were added.
- **Operational buffering**: at α≥0.9 unmet needs reach 10.0–10.3 in the
  control but only **2.5–3.1** in the main model; exploiter relief precision
  0.22 vs **0.37**; explorer precision 0.99 → 0.21 (control) vs 0.99 → 0.63
  (main). Network-bounded access both preserves the social chamber (F1) *and*
  buffers the operational collapse.

## 5. Finding 4 — Periphery

- Main model: the **spatial periphery MAE gap** (far − near quartile) widens
  with α: +0.13 (α=0) → **+0.32** (α=1.0); the aid contribution gap widens
  −1.8 → **−6.6** tokens per window. Control: both ≈ 0 at all α — the
  structural-null design holds.
- Network-periphery (betweenness) and broker gaps remain small (|gap| ≤ 0.06
  MAE) — as in the legacy run, the periphery story is primarily *spatial*
  under mobility, not graph-positional.

## 6. Ablation — the confirmation target is behaviourally near-inert

Seed-paired deltas (consensus − individual, fixed code) are statistically
indistinguishable from zero for SECI (both types), MAE, AI query share, AI
trust, lock-in and pool size, at every α, in both configurations. The three
isolated CI exclusions out of ~66 contrasts are small, incoherent, and include
one *opposite-signed* case (ΔSECI_exploit **+0.088** at α=0.7, main model:
consensus targeting made the exploiter chamber *shallower*). α* spread:
consensus [0.2–0.4] vs individual [0.3–0.6], both interior.

Why: the network consensus differs from the caller's own prior on only
~10–18% of exploiter-reported cells (≈1.5 levels when it does), is no more
accurate, and exploiters score confirmation against their *own stored prior*
regardless of what the AI reports. Consequences: (i) every headline finding is
invariant to the confirmation target, so the epistemically defensible
type-agnostic rule costs nothing; (ii) all differences between the fixed and
legacy runs (mainly overall AI-usage levels, ≈ +0.1 share) trace to the
reward-channel fixes, not to the targeting.

## 7. What changed vs the legacy run (for transparency in the SI)

| Quantity (main model) | Legacy | Fixed | Attribution |
|---|---|---|---|
| Combined SECI α=0 → 0.9 | −0.229 → −0.329 | −0.208 → −0.299 | unchanged within seed noise |
| Explorer MAE α=0 → 1.0 | 0.56 → 1.72 | 0.55 → 1.75 | unchanged |
| Exploiter AI share α=0 → 1.0 | 0.34 → 0.60 | 0.47 → 0.70 | Q-batching fix (level shift, same gradient) |
| Unmet needs α=1.0 (control / main) | 11.3 / 3.3 | 10.3 / 3.1 | unchanged |
| Interior α* spread (main) | [0.1–0.8] | [0.3–0.6] | tighter; reward fixes |

## 8. Open items before submission (details: PNAS_WRITING_INSTRUCTIONS §2, M1–M8)

1. **M1 — Metric revision (highest priority).** AECI-Var is judged
   inadequate as the AI-side counterpart of SECI: it is blind to the
   individualized (own-prior) bubble and is *most* negative at the truthful
   endpoint (truth-convergence confound) — it is not α-monotone in the
   direction the construct claims. Replacement: **AECI-IE**, SECI's exact
   variance-ratio formula applied to the report levels community members
   *receive from the AI channel* per window (with SECI-IE over the human
   channel as consistency check). Observation-only → the re-run reproduces
   all existing series on the same seeds and becomes the final citable
   dataset (`results-final/`).
2. **M2** cognitive-gap sweep on the fixed model (Fig. 3b). **M3** N=50
   boundary cells α∈{0.8, 0.9, 1.0}. **M4** mixed-model regressions with
   Holm-corrected contrasts.
3. **M5** robustness envelope (population, topology, AI supply,
   verification probability). **M6** dyadic docking (Glickman–Sharot).
4. **M7** salience decision experiment — keep `salience_weight=0` mainline
   (C12 as a *finding*) or promote a salience>0 variant; Finding 3's text
   depends on this author decision.
5. **M8** final figures from `results-final/`; Zenodo DOI; resolve the
   flagged placeholder reference; Significance Statement approval.

Full provenance and per-directory READMEs: `results/README.md`,
`results-verification/README.md`, `results-ablation-consensus/README.md`.

## 9. Status update (2026-08-21) — M1–M7 implemented, runs completed

**Code (all on `claude/pnas-paper-implementation-aqryq7`, merged through
PR #77 plus follow-ups).** M1 metrics (AECI-IE, SECI-IE, the channel- and
community-relative variants), the co-evolution series (AI reliance share,
delivered/effective α), M4 `tools/sweep_regression.py`, M6
`experiments/dyadic_docking.py`, `tools/coevolution.py`, and the
M2/M3/M5/M7 workflows and CLI flags. All metrics are observation-only:
trajectories verified bit-identical on pinned seeds before and after.

**Runs.**

| Purpose | Run | Status |
|---|---|---|
| N=20 paired sweep, M1 columns | 32404133354 | success; superseded |
| **N=20 paired sweep + co-evolution columns** | **32412891328** | **success — `results-final` candidate** |
| M2 cognitive-gap sweep (132 cells, main model) | 32425907960 | success (Fig. 3b input) |

**Author actions to close the loop.**

1. Archive run **32412891328** as `results-final/` and run **32425907960**
   as `results-gap-sweep-fixed/` (*Archive Run Artifacts*, `dest=` each).
   Artifact/CDN hosts are unreachable from the sandboxed agent
   environment, so archiving is the only route by which those numbers
   become extractable in-repo.
2. Extract the **per-type AECI-IE endpoints** from `results-final/`
   (`tools/coevolution.py`) and fill the five `\fillnum{}` placeholders in
   `DisasterAIFilter_PNAS.tex`. The compare workflow reports AECI-IE
   type-averaged only, and type-averaging cancels the two type-specific
   trends (see `docs/development/COEVOLUTION_ANALYSIS.md`).
3. **M1 metric decision** (`docs/development/M1_VALIDATION.md` §4) and
   **M7 salience decision**; both change manuscript text.
4. **M3** (N=50 boundary) and **M5** (robustness envelope) are dispatchable
   now that the workflows are on `main`.
5. **M8** figure regeneration: Figs. 2–4 need per-type panels from
   `results-final/`; Fig. 2 and Fig. 3a require new panel layouts
   (`comparison_configs_pertype.png`, `goldilocks_pnas.png`).

**Two findings from validation that affect the text** (details in
`docs/development/`):

- The belief-baseline AECI-IE collapses onto SECI for exploiters at
  α ≥ 0.8 (per-seed r = 0.89–0.95), so it is not an independent AI-side
  measurement there. Report it per type; the exploiter handover claim
  rides on AECI-IE-rel → 0 plus flat-high reliance plus flat MAE.
- Integer rounding of served reports makes delivered confirmation a step
  function of α that saturates at α ≥ 0.9 (α=0.9 and α=1.0 are identical
  at the report level). The operational cliff coincides with this
  boundary. Probabilistic rounding would linearize the dose but changes
  trajectories — an author decision, not applied.
