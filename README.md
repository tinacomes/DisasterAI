# DisasterAI

Agent-based simulation of disaster response under varying AI alignment levels. The model investigates whether there exists a "Goldilocks" alignment — a level at which AI is informative enough to break social echo chambers without creating AI-enforced filter bubbles.

## Research Question

AI systems in disaster response can be calibrated along a spectrum from *fully truthful* (reporting ground-truth severity) to *fully confirming* (echoing the human querier's own beliefs). We ask: what alignment level α minimises both social echo chambers (measured by SECI) and AI-induced filter bubbles (AECI) while preserving belief accuracy (MAE)?

## Model Overview

A 30 × 30 grid represents a disaster-affected area. 100 human agents update beliefs about local severity, seek information from peers and AI, and dispatch relief tokens. Five AI agents respond to queries with reports whose truth content is governed by α.

**Two agent types:**

| | Exploitative | Exploratory |
|---|---|---|
| Acceptance window D | 2.0 (narrow) | 4.0 (wide) |
| Acceptance steepness δ | 3.5 (sharp) | 1.2 (gradual) |
| Strategy | Confirmation-seeking; targets believed epicentre | Accuracy-seeking; targets high-uncertainty cells |
| Trust learning | Slow (lr = 0.015) | Faster (lr = 0.030) |

**AI alignment formula:**

```
r_AI = (1 − α) × truth + α × agent_belief
```

α = 0 → fully truthful; α = 1 → pure echo.

## Key Metrics

| Metric | Definition |
|---|---|
| **SECI** | Social Echo Chamber Index: 1 − Var(friend beliefs) / Var(global beliefs). Negative = echo chamber. |
| **AECI** | AI Echo Chamber Index: fraction of accepted updates from AI sources in a 5-tick window. |
| **MAE** | Mean absolute belief error on non-zero severity cells. |
| **α\*(bubble)** | argmin(‖SECI‖\_norm + ‖AECI‖\_norm) across the alignment sweep. |
| **α\*(+MAE)** | argmin(‖SECI‖\_norm + ‖AECI‖\_norm + MAE\_norm). |
| **Coverage deficit** | Mean (actual severity − aid tokens) per cell; positive = under-served. |

## Repository Structure

```
DisasterAI/
├── DisasterAI_Model.py       # Core ABM: agents, network, belief update, AI alignment
├── test_filter_bubbles.py    # Primary experiment: Goldilocks alignment sweep + all plots
├── simulate.py               # Lightweight single-run script
├── plot_results.py           # Aggregation and plotting for simulate.py outputs
├── requirements.txt          # Python dependencies
├── test_results/             # Output figures (gitignored; generated locally or via CI)
├── METHODS_PAPER.md          # Paper methodology section (main text)
└── SUPPLEMENTARY.md          # Supplementary tables and experimental design
```

## Installation

```bash
git clone <repo-url> && cd DisasterAI
pip install -r requirements.txt
```

Requires Python ≥ 3.11, Mesa ≥ 3.0, NumPy, NetworkX, Matplotlib, SciPy.

## Running Experiments

### Full Goldilocks sweep (N=20 replications per α level, ~30–60 min)

```bash
python test_filter_bubbles.py --n-runs 20 --collect-and-plot
```

Results are cached to `agent_model_results/`. Re-running with `--collect-and-plot` overwrites the cache; use `--plot-only` to regenerate figures without rerunning.

### Plot only (from existing JSON cache)

```bash
python test_filter_bubbles.py --plot-only
```

### Cognitive gap sweep

The 2D gap sweep (4g × 3 d\_mid × 11α = 132 conditions, N=20 seeded replications each)
is run via GitHub Actions:

- **GitHub → Actions → "Run Gap Sweep only"** — launches 132 parallel jobs, uploads
  per-cell JSON artifacts, then assembles `gap_sweep.png`
- **GitHub → Actions → "Replot Gap Sweep"** — re-generates `gap_sweep.png` from
  existing artifacts without re-running simulations (requires the source run ID)

### Single quick run

```bash
python simulate.py --alpha 0.5 --ticks 200
```

## Output Figures

All figures are saved to `analysis_plots/`.

| File | Description |
|---|---|
| `goldilocks_sweep.png` | Primary result: SECI, AECI, MAE, coverage deficit, and composite scores across α. Boxplots over N replications. Both α* values marked. |
| `timeseries_overview.png` | Time-series evolution of SECI, AECI, MAE, unmet needs, and AI query ratio for each α. |
| `echo_chamber_lifecycle.png` | SECI/AECI trajectories + bar charts showing echo-chamber formation and recovery timing per α level. |
| `aeci_evolution.png` | Cumulative AI query ratio over simulation time, separately for exploitative and exploratory agents. |
| `spatial_coverage.png` | Coverage deficit and aid density maps for α = 0 (baseline), α*(+MAE), and α*(bubble). |
| `periphery_gap.png` | Spatial near/far and network degree periphery gaps across α. |
| `gap_sweep.png` | Effect of cognitive polarisation (gap scalar g) on optimal α and composite scores. |

## Experimental Design Summary

### Primary: Goldilocks Alignment Sweep

- α ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}, N = 20 replications each
- 200 simulation ticks; steady-state window = last 75 ticks
- Total: 220 simulation runs

### Secondary: Cognitive Gap Sweep (2D)

- Gap scalar g ∈ {0.0, 0.5, 1.0, 1.5} × acceptance midpoint d\_mid ∈ {2.0, 3.0, 4.0} × α at 11 levels
- N = 20 independently seeded replications per cell, T = 200 ticks (2640 total runs)
- Tests robustness of α\* to both the degree of cognitive polarisation and the absolute level of cognitive openness
- α\* = 0.8 confirmed across all 132 (g, d\_mid) conditions

### Factor Sensitivity Sweeps (at fixed α = 0.5)

- Share of exploitative agents: {0.2, 0.5, 0.8}
- Disaster dynamics mode: {0 = static, 2 = moderate, 3 = high volatility}
- Rumour probability: {0.0, 0.5, 1.0}

## Interpreting Results

**SECI < 0** means friends hold more homogeneous beliefs than the full population — a social echo chamber. Very negative SECI at α = 1.0 reflects pure confirmation creating community convergence; near-zero SECI at α = 0.0 suggests truthful AI diversifies beliefs.

**AECI near 1** means agents are routing nearly all accepted updates through AI, creating AI-enforced homogeneity even if SECI is low. The Goldilocks zone is where both indices are simultaneously low.

**The spatial coverage maps** show averaged relief distribution (no epicentre marker, as epicentre locations vary randomly across replications). Red cells in the deficit map are chronically under-served; the near/far periphery gap quantifies whether distant cells suffer disproportionately.

## Citation

If you use this model, please cite [paper citation to be added].

## Licence

MIT — see `LICENSE`.
