# DisasterAI

Agent-based simulation of disaster response under varying AI alignment levels. The model investigates whether there exists a "Goldilocks" alignment — a level at which AI is informative enough to break social echo chambers without creating AI-enforced filter bubbles — and under which structural conditions that optimum exists at all.

## Research Question

AI systems in disaster response can be calibrated along a spectrum from *fully truthful* (reporting ground-truth severity) to *fully confirming* (echoing the querier's beliefs). We ask: what alignment level α minimises social echo chambers (SECI) and AI-induced filter bubbles (AECI-Var) while preserving belief accuracy (MAE) and relief performance — and does the answer depend on whether information access is bounded by the social network and agents' physical mobility?

## Model Overview

A 30 × 30 grid represents a disaster-affected area. 100 human agents update beliefs about local severity, seek information from peers and AI, and dispatch relief tokens. Five AI agents sense 15 % of the grid per tick (constant ±1 noise with p = 0.1, independent of α) and respond to queries with reports whose truth content is governed by α.

**Two agent types:**

| | Exploitative | Exploratory |
|---|---|---|
| Acceptance window D | 2.0 (narrow) | 4.0 (wide) |
| Acceptance steepness δ | 3.5 (sharp) | 1.2 (gradual) |
| Strategy | Confirmation-seeking; targets believed epicentre | Accuracy-seeking; targets high-uncertainty cells |
| Trust learning | Slow (lr = 0.015) | Faster (lr = 0.030) |

**AI alignment formula:**

```
r_AI = (1 − α) × truth + α × belief_target
```

α = 0 → fully truthful; α = 1 → pure echo. For exploitative queriers the confirmation target is the *trusted-network consensus* belief (community narrative); for exploratory queriers it is the individual prior. See METHODS_PAPER.md for the full mechanism, including the AI's deliberate overconfidence on unsensed cells.

**Three mechanism switches** (all default to baseline) define the two studied configurations:

| Switch | Baseline | Switched |
|---|---|---|
| `mobility` | 0 — agents immobile | 1 — home-anchored movement: exploiters are *returners* (r = 3), explorers roam to uncertain cells (r = 8) |
| `network_type` | `components` — disconnected type-homogeneous communities | `spatial_bridged` — spatially embedded communities + weak-tie bridges (brokers) |
| `query_scope` | `global` — any human reachable (trust-weighted) | `network` — friends + 2-hop reach only; access follows edges |

The baseline serves as the structural null (position and network topology barely constrain information access); the switched configuration is the realistic arm where spatial and network periphery effects are structurally possible.

## Key Metrics

All signed echo indices share one convention: **negative = echo chamber**, 0 = null, positive = more-diverse-than-population.

| Metric | Definition |
|---|---|
| **SECI** | Social Echo Chamber Index: community belief variance vs global variance (L1+ beliefs), piecewise-normalised to [−1, +1]. Computed per stored community, averaged per agent type. |
| **AECI-Var** | Same variance formula and L1+ pool as SECI, grouping by AI reliance (within-type median split on accepted AI updates). Enters the Goldilocks composite. Caveat: cannot distinguish convergence on truth from convergence on shared error — read together with AECI-Err. |
| **AECI-Err** | Confidence-weighted belief-error split, AI-heavy vs AI-light halves. Negative = AI-heavy agents more confidently wrong. |
| **AECI-Acc** | Share of accepted belief updates coming from AI (0–1, unsigned reliance measure). |
| **MAE** | Mean absolute belief error over disaster cells (true severity ≥ 1). |
| **Unmet needs** | Cells at severity ≥ 3 receiving zero relief tokens per tick. |
| **Precision** | Fraction of relief tokens placed on cells at severity ≥ 3 (at placement time). |
| **α\*(bubble)** | argmin(\|SECI\|ₙ + \|AECI-Var\|ₙ) across the sweep. |
| **α\*(+MAE)** | argmin(\|SECI\|ₙ + \|AECI-Var\|ₙ + MAEₙ). |

Normalised composites are range-normalised **within one sweep** — never compare their values across configurations; cross-configuration comparisons use raw metrics with paired per-seed deltas (`tools/compare_configs.py`). An α\* sensitivity analysis across six composite variants is reported with every sweep.

## Headline Results (comparison run, 2026-07-10)

From the seed-paired baseline-vs-switched comparison ([run 29100134858](https://github.com/tinacomes/DisasterAI/actions/runs/29100134858), post-fix code):

1. **Network-bounded access is a precondition for AI-amplified social echo chambers.** Under the switched configuration SECI deepens sharply at high α (−0.32 at α ≥ 0.9); under the baseline's global query pool it *weakens* instead (−0.13 at α = 1). Paired ΔSECI is significantly negative for α ≥ 0.7.
2. **A Goldilocks optimum exists only in the switched configuration**, and there it is robust: α\* ∈ [0.3, 0.6] across all six composite variants (primary 0.6, operational +MAE 0.4). The baseline has no interior bubble optimum (α\* = 1.0 for bubble-only composites).
3. **The switched configuration is operationally more robust to a confirming AI**: significantly lower MAE and higher precision at essentially every α, and the α ≥ 0.9 relief collapse is buffered (unmet needs ~3.5 vs ~11; precision 0.50 vs 0.21).

The PNG/CSV/MD files in the repository root are the committed snapshot of the **switched-configuration** sweep from that run (`summary_table.md` carries its α\* sensitivity block).

## Repository Structure

```
DisasterAI/
├── DisasterAI_Model.py           # Core ABM: agents, networks, mobility, belief update, AI alignment
├── test_filter_bubbles.py        # Experiment driver: sweeps, metrics, plots, summary tables
├── simulate.py                   # Lightweight single-run script
├── plot_results.py               # Aggregation/plots for simulate.py outputs
├── tools/
│   ├── compare_configs.py        # Baseline-vs-switched comparison: tables + paired deltas + figure
│   └── compare_epsilon.py        # Epsilon-decay comparison
├── .github/workflows/            # CI experiment pipelines (see table below)
├── METHODS_PAPER.md / .tex       # Paper methods section (Markdown + LaTeX)
├── SUPPLEMENTARY.md / .tex       # Supplementary tables and design
├── FINAL_MECHANICS_REVIEW.md     # Pre-submission mechanism audit (limitations inventory)
├── REVIEW_PROTOCOL.md            # Review findings C1–C12 and fix log
├── DESIGN_PROPOSAL_NETWORK_MOBILITY.md  # Design doc for the three mechanism switches
├── *.png, summary_table.*, experiment_results.json  # Results snapshot (switched config, 2026-07-10)
└── requirements.txt              # Python dependencies
```

## Installation

```bash
git clone <repo-url> && cd DisasterAI
pip install -r requirements.txt
```

Requires Python ≥ 3.11, Mesa ≥ 3.0, NumPy, NetworkX, Matplotlib, SciPy.

## Running Experiments

Paper-scale sweeps run on GitHub Actions (parallel per-α workers; a serial local sweep takes many hours):

| Workflow (Actions tab) | Purpose |
|---|---|
| **Compare Baseline vs Network/Mobility Switches** | The headline experiment: 11 α × 2 configs, then comparison tables (CSV/MD), paired per-seed deltas, overlay figure |
| **Run Primary Alignment Sweep only** | One 11-α sweep; inputs expose ticks, n_runs, salience, epsilon decay, and the three mechanism switches |
| **Run Gap Sweep only** | 2D cognitive-polarisation sweep (g × d_mid × α) |
| **Replot Primary Sweep / Replot Gap Sweep** | Regenerate figures + summary tables from a previous run's artifacts, no re-simulation |
| **Compare Epsilon Decay (Primary Sweep)** | Constant-ε baseline vs annealed exploration |
| **Spatial Visualization (manual)** | Fixed-epicentre (15, 15) sweep for interpretable coverage maps |
| **Run DisasterAI Experiment (full, manual)** | Full pipeline incl. factor sweeps |
| **Smoke test (on push)** | Fast integrity check |

Sweep artifacts include every figure plus `summary_table.csv` / `summary_table.md` (per-α raw metrics with SEs, normalised composites, α\* sensitivity) and `experiment_results.json` for replotting. Per-α JSON artifacts expire after 7 days, plots/tables after 30 — download or commit anything you need to keep.

Local quick runs:

```bash
# Single simulation
python3 simulate.py --alpha 0.5 --ticks 200

# Small local sweep (3 replications, ~15 min) — writes to test_results/
python3 test_filter_bubbles.py

# Regenerate figures from an existing results file
python3 test_filter_bubbles.py --plots-only

# Compare two collected sweeps on raw metrics
python3 tools/compare_configs.py <dir-with-plots-config-*/> <save-dir>
```

## Interpreting Results

**SECI < 0**: community members hold more homogeneous beliefs than the population — a social echo chamber. **AECI-Var < 0**: AI-reliant agents are more homogeneous than the population — but at low α this partly reflects convergence *on the truth*; check AECI-Err (confidently-wrong beliefs) before calling it harm.

**Spatial coverage maps** from randomised-epicentre sweeps average 20 different disaster locations and are not interpretable as maps; use the fixed-epicentre Spatial Visualization workflow for map figures. The near/far periphery *statistics* are unaffected (computed per run relative to each run's own epicentre).

**Cross-configuration claims** must use raw metrics (paired per-seed deltas in `comparison_table.md`), never the within-sweep normalised composites.

## Citation

If you use this model, please cite [paper citation to be added].

## Licence

MIT — see `LICENSE`.
