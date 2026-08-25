# Results Compendium — single source of truth (2026-08-25)

One document that maps **every claim to its dataset, figure, and table**.
All headline numbers come from the current-model record
**`results-mechfix/`** (run 32821105202, commit `30f89e0`); everything else
is context, ablation, or counterfactual. Steady state = mean over the last
75 of 200 ticks; N = 20 seed-paired replications per condition; sign
convention: **negative SECI/AECI = echo chamber**. Main model ("switches")
= mobility + spatially embedded bridged communities + network-gated
queries; control ("baseline") = immobile + disconnected communities +
global query access.

Companion documents: [`RESULTS_OVERVIEW.md`](RESULTS_OVERVIEW.md)
(historical layers + §0 update), [`PNAS_WRITING_INSTRUCTIONS.md`](PNAS_WRITING_INSTRUCTIONS.md)
(how to revise the paper — updated with this compendium).

## 1. Dataset map

| Directory | Run | Code | Status / role |
|---|---|---|---|
| [`results-mechfix/`](results-mechfix/README.md) | 32821105202 | `30f89e0` | **CANONICAL** — revised model, baseline vs switches |
| [`results-ablation-ownref/`](results-ablation-ownref/README.md) | 32839355312 | `2b2ffd7` | Ablation A: legacy own-prior confirmation (network-reference effect) |
| [`results-ablation-detround/`](results-ablation-detround/README.md) | 32839367900 | `2b2ffd7` | Ablation B: legacy deterministic rounding (dose-linearisation effect) |
| [`results-salience/`](results-salience/README.md) | 32852587340 | `bbfb5ec` | M7 salience counterfactual (C12 + exploiter trap), **decided** |
| [`results-final/`](results-final/README.md) | 32412891328 | `36891da` | Pre-revision record (superseded; step-function dose) |
| [`results-ablation-consensus/`](results-ablation-consensus/README.md) | 32338797843 | `55d4b2b` | AI-side confirmation-target ablation (inert — SI transparency) |
| [`results-verification/`](results-verification/README.md) | 32298278561 | `55d4b2b` | Type-agnostic-fix verification (historical) |
| `results/` | 32040117179 | `0e05139` | Legacy record (SI transparency table only) |
| `results-gap-sweep-fixed/` | 32425907960 | pre-revision | M2 gap sweep — **needs re-run** under `30f89e0` |

## 2. Headline results (with evidence)

### H1 — There is a robust interior alignment optimum: α\* = 0.6 in the main model

α\* = 0.6 for **7 of 12** composite definitions (both population
composites, SECI+AECI-IE-chan ± MAE, SECI-only, retired AECI-Var ± MAE);
remaining variants 0.1–0.3. Control: bubble composites 0.8–0.9 (interior).
Operational optimum coincides: unmet needs 1.59 → **0.25 (α=0.6)** → 2.86
(main); 2.23 → 0.49 (α=0.6) → 10.18 (control).

- Figure: [`results-mechfix/plots-config-switches/goldilocks_alignment_sweep.png`](results-mechfix/plots-config-switches/goldilocks_alignment_sweep.png)
  and [`alpha_star_sensitivity.png`](results-mechfix/plots-config-switches/alpha_star_sensitivity.png)
- Table: α\* per composite per configuration —
  [`results-mechfix/config-comparison/comparison.txt`](results-mechfix/config-comparison/comparison.txt)
  (§"α\* sensitivity"); raw metrics
  [`comparison_table.csv`](results-mechfix/config-comparison/comparison_table.csv)

### H2 — The optimum's robustness is attributable to the dose linearisation

Single-mechanism ablations: reverting only the stochastic rounding
fragments α\* to spread [0.2, 0.6] (4/12 at 0.6); reverting only the
network confirmation reference changes nothing (8/12 at 0.6, all series
within seed noise). The delivered dose, not nominal α, is what curves
respond to; effective α under legacy rounding was a near step function.

- Tables: [`results-ablation-detround/.../summary_table.md`](results-ablation-detround/filter-bubble-primary-plots/summary_table.md),
  [`results-ablation-ownref/.../summary_table.md`](results-ablation-ownref/filter-bubble-primary-plots/summary_table.md)
- Verdict table: [`results-mechfix/README.md`](results-mechfix/README.md) §Attribution

### H3 — Network-bounded access is the structural precondition for the social chamber

Under network-gated access the exploitative communities' echo chamber
persists at every α (SECI_exploit −0.39 … −0.18); under global access it
dissolves (−0.45 → **+0.05**). The explorer chamber deepens 0 → −0.33 with
α in **both** configurations. Combined paired ΔSECI (main − control) at
α ≥ 0.7 is seed-robust (α=1: −0.140 [−0.211, −0.069]).

- Figure: [`results-mechfix/config-comparison/comparison_configs.png`](results-mechfix/config-comparison/comparison_configs.png);
  per-type panels in each config's `goldilocks_alignment_sweep.png` /
  `bubble_timeseries.png`
- Table: paired per-seed deltas —
  [`comparison.txt`](results-mechfix/config-comparison/comparison.txt) §"Paired per-seed deltas"

### H4 — Confirming AI harms accuracy-seekers by starvation

Explorer MAE 0.54 → 1.74 (main; 0.62 → 1.94 control) while exploiter MAE
is flat ≈1.8 at all α. Mechanism: the explorers' L1+ belief pool collapses
113 → 29 per agent (main; 108 → 19 control) — the confirming AI echoes
their mostly-empty priors, so they stop acquiring disaster knowledge. The
dose–response is smooth at every α step (no α=0.5 jump, no α=0.9 cliff).

- Figures: MAE panel of `goldilocks_alignment_sweep.png` (both configs);
  evolution in [`bubble_timeseries.png`](results-mechfix/plots-config-switches/bubble_timeseries.png)
- Table: per-α series incl. `l1pool_*` in each config's
  [`summary_table.csv`](results-mechfix/plots-config-switches/summary_table.csv)

### H5 — Exploiter capture is real — and contingent on base-rate-diluted evaluation

Main model: exploiter AI query share rises 0.54 → 0.69 with α (AI trust
flat ≈0.47–0.49). Mechanistic diagnosis (probe): at α=0 a truthful AI
*agrees* with an exploiter on most reported cells (empty periphery +
already-correct beliefs), so the batched confirmation reward is only
weakly negative — the trap is diluted, capture grows with α. The salience
counterfactual (s=1) removes the dilution: the capture gradient vanishes
(AI share flat ≈0.48–0.51, trust ≈0.42 at all α) — **but exploiters
retrench into their social network instead of becoming accurate**: at α=0
their chamber deepens (SECI −0.52 vs −0.37) and precision falls (0.43 vs
0.57), MAE unchanged.

- Figures: [`aeci_evolution.png`](results-mechfix/plots-config-switches/aeci_evolution.png)
  (query-share trajectories per type)
- Tables: [`results-salience/salience-tables/robustness_tables.md`](results-salience/salience-tables/robustness_tables.md);
  verdict in [`results-salience/README.md`](results-salience/README.md)

### H6 — C12 stands at every salience level (explorers never punish confirming AI via trust)

Explorer AI trust 0.88 → 0.84 across α at s=0; 0.90 → 0.82 even at full
salience — severity-weighted verification also rewards the AI's coverage
of disaster cells, so explorer AI use is *higher* under salience.
`salience_weight = 0` stays mainline; C12 is a finding, not an artifact.

- Table: `AItrust_er` column in
  [`robustness_tables.md`](results-salience/salience-tables/robustness_tables.md)

### H7 — Fragmentation and the societal layer are different answers

Population-level SECI in the control runs −0.22 → 0.00 across α — the
societal lens **masks** the crossing where the exploiter chamber
dissolves while the explorer chamber deepens. The population AI-channel
index (AECI-IE-chan-pop) is **U-shaped in α in both configurations**
(shallowest ≈ −0.15/−0.19 at α=0.7): society's served information is most
diverse at intermediate alignment, matching the operational optimum. The
population composites are the most ablation-robust α\* definition (0.6 in
all three revised-model datasets).

- Figure: [`population_evolution.png`](results-mechfix/plots-config-switches/population_evolution.png)
  (3×3: SECI / AECI-IE-chan / LockIn × population / exploit / explor —
  full trajectories, so lifecycle dynamics are not aggregated away)
- Tables: population composite rows in
  [`comparison.txt`](results-mechfix/config-comparison/comparison.txt)

### H8 — Network-bounded access buffers the operational collapse; harms concentrate on the spatial periphery

At α=1: unmet needs 2.86 (main) vs 10.18 (control); explorer precision
0.62 vs 0.20; exploiter precision 0.38 vs 0.26. The main model dominates
the control on MAE and precision at *every* α (paired CIs exclude 0).
Periphery: spatial MAE gap (far − near) widens 0.12 → 0.33 and the
aid-contribution gap −1.8 → −6.6 with α in the main model; both ≈ 0 in the
immobile control (structural null holds under the revised model).

- Figures: [`periphery_gap.png`](results-mechfix/plots-config-switches/periphery_gap.png),
  [`periphery_gap_evolution.png`](results-mechfix/plots-config-switches/periphery_gap_evolution.png),
  unmet-needs panel of `goldilocks_alignment_sweep.png`
- Table: paired deltas (Unmet, Precision, MAE) in
  [`comparison.txt`](results-mechfix/config-comparison/comparison.txt)

## 3. Metric guide (what to cite for what)

| Construct | Metric | Status |
|---|---|---|
| Social echo chamber | **SECI** per type + **SECI-pop** | Primary (belief-based) |
| AI-side echo, comparable to SECI | **AECI-IE-chan** per type + **-pop** (channel baseline) | Primary AI-side index |
| Individualised AI lock-in | **AECI-LockIn** per type + L1+ pool | Mechanism evidence |
| Accuracy / harm | MAE per type (disaster cells), precision, unmet needs | Operational outcomes |
| AECI-IE (belief baseline) | explorer dose–response in the control only | Secondary; confounded for exploiters (M1) |
| AECI-Var, AECI-Err, query-share AECI | retired / secondary | α\* sensitivity table only |
| Dose manipulation check | `effective_alpha` | Cite once for the rounding rationale |

## 4. What must still (re-)run before submission

1. **M2 re-run — DONE**
   ([`results-gap-sweep-mechfix/`](results-gap-sweep-mechfix/README.md),
   run 32866377905, all 132 cells): the interior optimum holds in **every**
   (g, d_mid) cell (unmet α\* = 0.6–0.7; population bubble α\* = 0.6–0.8;
   explorer MAE ≈0.55→1.75 everywhere). The intended "optimum location
   depends on the cognitive profile" claim is NOT supported at the
   population/operational level — reframe Fig. 3b as a **robustness**
   result (the optimum is structural, not a knife-edge of the D/δ mix);
   only the per-type channel composite drifts with the gap (suggestive,
   needs M4 CIs).
2. **M3** — N=50 boundary cells (α ∈ {0.8, 0.9, 1.0}) on the revised model.
3. **M4** — mixed-model regressions + Holm-corrected contrasts
   (`tools/sweep_regression.py`) over `results-mechfix/`.
4. **M5** — robustness envelope (population, topology, AI supply,
   verification probability): `docs/robustness_summary.md` is still
   **PENDING RUNS**.
5. **M6 — DONE, PASS** ([`results-docking/`](results-docking/README.md),
   run 32866369049): the dyad qualitatively reproduces Glickman & Sharot's
   human–AI amplification for both agent types (false belief corrected
   under truthful AI, fully preserved under confirming AI, smooth
   gradient; aligned AI retains more bias than a human partner). The
   external-validity anchor is in place; the venue escalation now hinges
   on M2–M5.

## 5. Documents to check (author)

1. [`results-mechfix/README.md`](results-mechfix/README.md) — canonical
   provenance + attribution verdict.
2. [`results-salience/README.md`](results-salience/README.md) — M7
   decision + the social-retrenchment finding (new Discussion material).
3. [`results-ablation-ownref/README.md`](results-ablation-ownref/README.md) /
   [`results-ablation-detround/README.md`](results-ablation-detround/README.md).
4. [`RESULTS_OVERVIEW.md`](RESULTS_OVERVIEW.md) §0 (update layer; §§1–8
   are the historical chain).
5. [`PNAS_WRITING_INSTRUCTIONS.md`](PNAS_WRITING_INSTRUCTIONS.md) —
   revised paper-writing guidance incl. the venue recommendation.
6. Key figures to eyeball:
   [`comparison_configs.png`](results-mechfix/config-comparison/comparison_configs.png),
   both configs' `goldilocks_alignment_sweep.png`,
   [`population_evolution.png`](results-mechfix/plots-config-switches/population_evolution.png).
7. Historical (context only, superseded claims corrected in place):
   [`results-final/README.md`](results-final/README.md),
   [`docs/development/M1_VALIDATION.md`](docs/development/M1_VALIDATION.md).
