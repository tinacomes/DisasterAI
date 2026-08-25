# Results Overview — Fixed Model + Mechanism Revision (2026-08-25)

Consolidated overview for author review. Steady state = mean over the last
75 ticks; N = 20 seed-paired replications per α level; 200 ticks per run.

## 0. Update 2026-08-25 — mechanism revision restores the interior optimum

Commit `30f89e0` revised two mechanisms (defaults; legacy behaviour behind
flags) and added a population-level metric layer; the new head-to-head sweep
is archived in **`results-mechfix/`** (run 32821105202), which supersedes
`results-final/` as the current-model record.

**What changed and why.** (1) `confirmation_reference='network'`:
exploiters' confirmation reward now targets the **trusted-network
consensus** (fallback: own stored prior). The previously documented
"confirm the social network" reference was dead code — the consensus fed
only the accuracy channel, which exploiters never use. (2)
`report_rounding='stochastic'`: probabilistic rounding of the AI's aligned
report makes the **delivered** confirmation dose linear in α; the legacy
`np.round` was a near step function (≈0 for α≤0.4, saturated for α≥0.9), so
half the sweep delivered almost no treatment and the α=0.9 "cliff" was a
delivery artifact. (3) `salience_weight` now also covers the exploiters'
confirmation channel (neutral at the mainline default 0). (4) New
population-level series (`seci_pop`, `aeci_ie_pop`, `lockin_pop`,
population composites, `population_evolution.png`): per-type series show
the **fragmentation** between cognitive styles; the population series
answer the **societal** question, and the full trajectories are plotted so
formation/dissolution dynamics are no longer hidden by the late-run window.

**Headline results (results-mechfix).**

- **Interior α\* is back and composite-robust in the main model**:
  α\* = **0.6** for 7 of 12 composites (including both population
  composites, SECI + AECI-IE-chan ± MAE, and SECI-only); remaining
  variants 0.1–0.3. Control: bubble composites 0.8–0.9 — the pre-revision
  corner solution (α\* = 1.0) is gone.
- **Smooth dose–response**: combined MAE rises gradually at every α step
  (1.17 → 1.82 main; 1.25 → 1.96 control); no α=0.5 jump, no α=0.9 cliff.
- **Per type**: the exploitative chamber persists at all α under
  network-bounded access (SECI_exploit −0.39 … −0.18) and dissolves under
  global access (−0.45 → +0.05); the explorer series deepens 0 → −0.33 in
  both configurations; explorer MAE 0.54 → 1.74 — Findings 1 and 2 carry
  over qualitatively with a cleaner gradient.
- **Population**: SECI-pop in the control runs −0.22 → 0.00, masking the
  exploiter-dissolve/explorer-deepen crossover (why both levels are
  reported). AECI-IE-chan-pop is **U-shaped in α in both configurations**
  (shallowest at α≈0.7): the served information environment is most
  diverse at intermediate alignment.
- **Operational U-shape intact**: unmet needs 1.59 → **0.25 (α=0.6)** →
  2.86 (main); 2.23 → 0.49 (α=0.6) → 10.18 (control). The switches
  configuration dominates the control on MAE and precision at every α
  (paired CIs exclude 0) and cuts unmet needs by −7.3 cells at α=1.

Attribution (settled by the single-mechanism ablations, 2026-08-25): the
**stochastic-rounding dose linearisation drives the restored interior α\***
— reverting only the rounding (`results-ablation-detround/`) fragments α\*
to spread [0.2, 0.6] (4/12 composites at 0.6), while reverting only the
network confirmation reference (`results-ablation-ownref/`) leaves the
result essentially unchanged (8/12 at 0.6, all series within seed noise).
The network reference matters for mechanism-(ii) coherence, not for the
aggregate curves. Population composites are α\* = 0.6 in all three
datasets — the most ablation-robust definition.
Sections 1–8 below describe the pre-revision record and remain valid as
the ablation/history chain; numbers there refer to `results-verification/`
unless stated.

## 1. The runs

| Run | Directory | Code | AI confirmation target | Purpose |
|---|---|---|---|---|
| 32040117179 (2026-08-17/18) | `results/` | `0e05139` (legacy) | consensus for exploitative callers | Legacy record; basis of the NHB draft |
| 32298278561 (2026-08-19) | `results-verification/` | `55d4b2b` (fixed) | caller's own prior, both types | Verification of the type-agnostic fix |
| 32338797843 (2026-08-20) | `results-ablation-consensus/` | `55d4b2b` (fixed) | consensus (legacy rule re-enabled) | Ablation isolating the AI confirmation target |
| 32412891328 (2026-08-20) | `results-final/` | `36891da` | caller's own prior | Pre-revision record + M1 metrics (superseded) |
| **32821105202 (2026-08-25)** | **`results-mechfix/`** | **`30f89e0`** | **caller's own prior** (exploiters *evaluate* vs network consensus) | **Current-model record — candidate canonical dataset** |

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
- **Interior α\*** (verification run, AECI-Var composites): in the main
  model all six composite definitions gave an interior optimum, spread
  **[0.3, 0.6]**. *Caution:* this statement did **not** survive the switch
  to the AECI-IE primary composites in `results-final/` (α\* jumped to the
  corners: 1.0 control / 0.0 main), which was one trigger for the
  2026-08-25 mechanism revision. Under the revised model
  (`results-mechfix/`) the interior optimum is restored and
  composite-robust: α\* = 0.6 for 7/12 composites in the main model — see
  §0. Use §0's numbers for the paper.

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
accurate, and — in the pre-revision code — exploiters scored confirmation
against their *own stored prior* regardless of what the AI reports.
Consequences: (i) every headline finding is invariant to the AI-side
confirmation target, so the epistemically defensible type-agnostic rule
costs nothing; (ii) all differences between the fixed and legacy runs
(mainly overall AI-usage levels, ≈ +0.1 share) trace to the reward-channel
fixes, not to the targeting.

*Terminology guard:* this ablation concerns the **AI's** targeting
(`confirmation_target`, individual vs consensus — what the AI serves). It
is distinct from the 2026-08-25 `confirmation_reference` revision, which
concerns the **exploiters' own evaluation** (what they score confirmation
against). The AI remains type-agnostic and individual-targeting throughout.

## 7. What changed vs the legacy run (for transparency in the SI)

| Quantity (main model) | Legacy | Fixed | Attribution |
|---|---|---|---|
| Combined SECI α=0 → 0.9 | −0.229 → −0.329 | −0.208 → −0.299 | unchanged within seed noise |
| Explorer MAE α=0 → 1.0 | 0.56 → 1.72 | 0.55 → 1.75 | unchanged |
| Exploiter AI share α=0 → 1.0 | 0.34 → 0.60 | 0.47 → 0.70 | Q-batching fix (level shift, same gradient) |
| Unmet needs α=1.0 (control / main) | 11.3 / 3.3 | 10.3 / 3.1 | unchanged |
| Interior α* spread (main) | [0.1–0.8] | [0.3–0.6] | tighter; reward fixes |

## 8. Open items before submission (details: PNAS_WRITING_INSTRUCTIONS §2, M1–M8)

1. **M1 — Metric revision: status after `results-mechfix/`.** AECI-Var
   remains retired (blind to the individualized bubble; truth-convergence
   confound at α=0). The belief-baseline **AECI-IE** did not deliver the
   intended dose–response at N=20 in the pre-revision run (explorer series
   ≈0 in the main model; exploiter series anti-monotone) and its magnitude
   is aggregation-fragile (NaN-ragged 5-tick pools). Under the revised
   model the workable AI-side indices are: explorer AECI-IE in the control
   (+0.03 → −0.36, ≈0 at the truthful endpoint), the **channel-baseline
   AECI-IE-chan per type**, and the **population AECI-IE-chan-pop**
   (U-shaped in α, both configurations). Recommended primary for the
   paper: the channel-baseline pair (per-type + population), with
   AECI-LockIn and the L1+ pool as the individual-lock-in evidence.
2. **M2** cognitive-gap sweep on the fixed model (Fig. 3b). **M3** N=50
   boundary cells α∈{0.8, 0.9, 1.0}. **M4** mixed-model regressions with
   Holm-corrected contrasts.
3. **M5** robustness envelope (population, topology, AI supply,
   verification probability). **M6** dyadic docking (Glickman–Sharot).
4. **M7** salience decision experiment — keep `salience_weight=0` mainline
   (C12 as a *finding*) or promote a salience>0 variant; Finding 3's text
   depends on this author decision.
5. **M8** final figures from `results-mechfix/` (or its successor once the
   single-mechanism ablations are in); Zenodo DOI; resolve the flagged
   placeholder reference; Significance Statement approval.
6. **New (2026-08-25):** single-mechanism ablations of the revision
   (`confirmation_reference=own`; `report_rounding=deterministic`) to
   attribute the restored interior α\*; optional `salience_weight>0` sweep
   now that salience covers both agent types' channels.

Full provenance and per-directory READMEs: `results/README.md`,
`results-verification/README.md`, `results-ablation-consensus/README.md`,
`results-final/README.md`, `results-mechfix/README.md`.
