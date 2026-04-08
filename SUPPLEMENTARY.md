# Supplementary Materials

## S1. Model Parameters

**Table S1. Fixed simulation parameters (all experiments).**

| Parameter | Symbol | Value | Rationale |
|---|---|---|---|
| Grid width / height | — | 30 × 30 | Sufficient spatial resolution; tractable for N=20 replications |
| Number of human agents | N | 100 | Large enough for network effects; small enough to track individually |
| AI agents | — | 5 | ~1 AI per 20 humans; ensures non-negligible but not dominant coverage |
| AI sensing fraction (per tick) | — | 0.15 | AI sees 15% of the 900-cell grid ≈ 135 cells/tick |
| Share of exploitative agents | — | 0.50 | Balanced population; varied in factor sweep (see Table S3) |
| Share of confirming social ties | — | 0.70 | Reflects empirically documented homophily in disaster communication |
| Disaster dynamics mode | — | 2 | Medium-tempo evolution: cells drift ±1–2/tick toward Gaussian baseline |
| Shock probability (per cell/tick) | — | 0.10 | Persistent environmental non-stationarity |
| Shock magnitude | — | ±2 levels | Produces salient deviations requiring active information-seeking |
| Simulation length | T | 200 ticks | Allows learning, echo-chamber formation, and steady-state measurement |
| Initial (human) trust | — | 0.30 | Low baseline prevents unrealistic blind trust at t=0 |
| Initial AI trust | — | 0.25 | Slight AI skepticism, consistent with survey findings on AI acceptance |
| Q-learning rate | η | 0.10 | Standard value; sensitivity unreported here |
| ε-greedy exploration | ε | 0.30 | Ensures continued source diversity throughout run |
| Trust learning rate (exploitative) | — | 0.015 | Slow trust revision matches confirmation-seeking profile |
| Trust learning rate (exploratory) | — | 0.030 | Faster but still sub-linear trust revision |
| Rumour probability | — | 1.00 | All social information propagated (worst-case echo scenario) |
| Relief outcome delay | — | 15–25 ticks | Random; models logistical pipeline in humanitarian response |
| Steady-state window | — | 75 ticks | Last 75 of 200 ticks (15 metric samples × 5-tick cadence) |

---

## S2. Agent Cognitive Parameters

**Table S2. Bayesian acceptance parameters by agent type.**

| Parameter | Exploitative | Exploratory | Interpretation |
|---|---|---|---|
| Acceptance half-width, D | 2.0 | 4.0 | Distance at which P(accept) = 0.5 |
| Acceptance steepness, δ | 3.5 | 1.2 | Higher δ → sharper threshold |
| Precision weight | 1.5 × c/(1−c) | 0.8 × c/(1−c) | Resistance to belief revision (c = confidence) |
| Sensing radius | 0 cells | 0 cells | Both types rely on communication (no direct distant sensing) |
| Interest point targeting | Believed epicentre | Highest-uncertainty cell | Governs where queries are directed |

The acceptance probability for a report $r$ given prior belief $b$ is:

$$P(\text{accept}) = \frac{D_\text{eff}^\delta}{|r - b|^\delta + D_\text{eff}^\delta}$$

where $D_\text{eff} = D \cdot (1 + 0.5\, T_\text{source})$ for social contacts (trust-scaled) and $D_\text{eff} = D$ for AI sources. This generalises the bounded-confidence model (Deffuant et al. 2000) by replacing the binary threshold with a smooth logistic-type curve parameterised by (D, δ).

---

## S3. AI Alignment Mechanism

The AI response formula is:

$$r_\text{AI} = (1 - \alpha)\, t + \alpha\, b$$

where $t$ is the ground-truth cell severity, $b$ is the querying agent's current belief, and α ∈ [0, 1] is the alignment parameter. Key properties:

- At α = 0: $r_\text{AI} = t$ — fully truthful report.
- At α = 1: $r_\text{AI} = b$ — perfect confirmation of the agent's existing belief.
- Intermediate α: weighted average; the AI is partially informative and partially sycophantic.

This linear interpolation is the simplest formulation that spans the truth–confirmation spectrum without introducing additional free parameters. Non-linear or stochastic variants (e.g. $r \sim \mathcal{N}(\alpha b + (1-\alpha)t,\, \sigma^2)$) were not pursued in the main sweep to preserve interpretability; adding noise would shift all SECI/AECI curves downward without changing the location of α* as long as noise is symmetric.

---

## S4. Cognitive Gap Scalar

The gap scalar g parameterises the cognitive divergence between agent types around a shared midpoint:

$$D_\text{exploit}(g) = \max(3.0 - 1.0\,g,\; 0.5), \quad \delta_\text{exploit}(g) = 2.35 + 1.15\,g$$
$$D_\text{explor}(g) = 3.0 + 1.0\,g, \quad \delta_\text{explor}(g) = \max(2.35 - 1.15\,g,\; 0.1)$$

**Table S4. Cognitive parameters at each gap scalar value.**

| g | D\_exploit | δ\_exploit | D\_explor | δ\_explor |
|---|---|---|---|---|
| 0.0 | 3.00 | 2.35 | 3.00 | 2.35 |
| 0.5 | 2.50 | 2.93 | 3.50 | 1.78 |
| 1.0 | 2.00 | 3.50 | 4.00 | 1.20 |
| 1.5 | 1.50 | 4.08 | 4.50 | 0.63 |

The shared midpoint (D\_mid = 3.0, δ\_mid = 2.35) is the arithmetic mean of the baseline exploit/explor values. At g = 0 both types are cognitively identical, providing a null condition for heterogeneity effects. The invariant D\_exploit < D\_explor and δ\_exploit > δ\_explor is maintained for all g > 0, ensuring that the exploitative type always has a narrower and steeper acceptance curve than the exploratory type.

The gap sweep extends the design to a second axis: d\_mid ∈ {2.0, 3.0, 4.0} independently varies the absolute level of cognitive openness (closed, baseline, open) orthogonally to the inter-type gap g. The full 2D sweep uses N = 20 independently seeded replications per (g, d\_mid, α) cell and T = 200 ticks, matching the primary alignment sweep.

---

## S5. Metrics

### S5.1 Social Echo Chamber Index (SECI)

$$\text{SECI} = 1 - \frac{\text{Var}\bigl(\{b_{f,k} : f \in \text{friends}(i),\, k \in \text{cells}\}\bigr)}{\text{Var}\bigl(\{b_{j,k} : j \in \text{all agents},\, k \in \text{cells}\}\bigr)}$$

Computed separately for exploitative and exploratory agents; reported as the mean across all agents of that type. A formation event is registered when SECI < −0.30 sustained for ≥ 3 consecutive samples; a recovery event when SECI ≥ 0.0 sustained for ≥ 5 consecutive samples after formation. If formation never occurs the lifecycle bar is left blank (○); if formation occurs but recovery does not by the end of the run the bar reaches the maximum height with annotation ✗.

### S5.2 AI Echo Chamber Index (AECI)

$$\text{AECI} = \frac{\Delta\, q_\text{AI}}{\Delta\, q_\text{AI} + \Delta\, q_\text{human}}$$

where Δ*q* denotes the increment in accepted information-update calls over a 5-tick window. This window-delta formulation distinguishes recent reliance from cumulative history, making AECI responsive to within-run behavioural shifts.

### S5.3 Belief Accuracy (MAE)

$$\text{MAE} = \frac{1}{|\mathcal{C}|} \sum_{k \in \mathcal{C}} |b_{i,k} - t_k|$$

averaged over agents $i$ and over the set $\mathcal{C}$ of cells with true severity $t_k \geq 1$.

### S5.4 Composite Goldilocks Scores

Let $\bar{m}(q, \alpha)$ denote the steady-state mean (last 75 ticks) of metric $m$ at alignment level α. Range-normalisation across the sweep:

$$m_\text{norm}(\alpha) = \frac{\bar{m}(\alpha) - \min_\alpha \bar{m}}{\max_\alpha \bar{m} - \min_\alpha \bar{m}}$$

Then:

$$\text{total\_bubble\_norm}(\alpha) = |\text{SECI}|_\text{norm}(\alpha) + |\text{AECI}|_\text{norm}(\alpha)$$
$$\text{total\_score\_norm}(\alpha) = |\text{SECI}|_\text{norm}(\alpha) + |\text{AECI}|_\text{norm}(\alpha) + \text{MAE}_\text{norm}(\alpha)$$

$$\alpha^* = \arg\min_\alpha \text{total\_bubble\_norm}(\alpha)$$
$$\alpha^*(+\text{MAE}) = \arg\min_\alpha \text{total\_score\_norm}(\alpha)$$

Range normalisation ensures each component contributes on the same [0, 1] scale regardless of the absolute magnitude of SECI, AECI, or MAE. The two optima are reported separately so that the trade-off between echo-chamber suppression and accuracy improvement is explicit.

### S5.5 Spatial and Network Periphery Metrics

**Spatial periphery.** For each simulation run, cells are ranked by Euclidean distance to that run's epicentre. Near cells (Q1, closest quartile) and far cells (Q4, furthest quartile) are identified per run; coverage deficit (actual severity − mean aid tokens per tick) and aid density are averaged within each set. These per-run scalars are then averaged across replications. This per-run computation avoids the blurring artefact that would arise from mapping all replications onto a single grid when epicentre locations vary.

**Network periphery.** Agents are ranked by their Watts–Strogatz degree. Low-degree agents (Q1) and high-degree agents (Q4) are compared on belief MAE and relief tokens sent per tick.

---

## S6. Primary Experiment: Goldilocks Alignment Sweep

**Table S6. Experiment S6 design.**

| Dimension | Values | N per cell |
|---|---|---|
| AI alignment α | 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 | 20 |
| Agent type mix | 50% exploit / 50% explor (fixed) | — |
| Simulation length T | 200 ticks | — |
| **Total simulation runs** | **220** | — |

Each replication uses an independent random seed for epicentre placement, agent initialisation, and shock sequences. Results are presented as boxplots (median, IQR, 1.5 × IQR whiskers) over the 20 replications. Steady-state scalars are averaged over the last 75 ticks (15 samples at the 5-tick cadence).

---

## S7. Secondary Experiment: Cognitive Gap Sweep

**Table S7. Experiment S7 design.**

| Dimension | Values | N per cell |
|---|---|---|
| Gap scalar g | 0.0, 0.5, 1.0, 1.5 | 20 |
| Acceptance-window midpoint d\_mid | 2.0 (closed), 3.0 (baseline), 4.0 (open) | — |
| AI alignment α (within each g, d\_mid) | 0.0, 0.1, …, 1.0 (11 levels) | 20 |
| Simulation length T | 200 ticks | — |
| **Total simulation runs** | **2640** (4 × 3 × 11 × 20) | — |

Each replication uses an independent random seed (seed = run index). The d\_mid axis tests whether the Goldilocks finding is robust to the absolute level of cognitive openness, independently of the inter-type gap g. Results confirm α\* = 0.8 across all 132 (g, d\_mid) conditions; the full N = 20 design is treated as confirmatory. Both α\*(bubble) and α\*(+MAE) are computed for each condition and shown in the figure (solid and dashed lines respectively).

---

## S8. Factor Sensitivity Sweeps

Three additional one-at-a-time factor sweeps test the robustness of the Goldilocks finding across environmental and population variation. All use α = 0.5 (the midpoint of the alignment range) and N = 2 replications, with run length max(50, T/2) = 125 ticks.

**Table S8. Factor sweep design.**

| Factor | Values swept | Fixed parameters |
|---|---|---|
| Share of exploitative agents | 0.2, 0.5, 0.8 | All base\_params except share\_exploitative |
| Disaster dynamics mode | 0 (static), 2 (moderate), 3 (high volatility) | All base\_params except disaster\_dynamics |
| Rumour probability | 0.0, 0.5, 1.0 | All base\_params except rumor\_probability |

Disaster dynamics mode controls the rate at which cell severities drift and receive shocks: mode 0 fixes the initial disaster map; mode 2 applies the baseline shock schedule (Table S1); mode 3 doubles shock frequency.

---

## S9. Software and Reproducibility

The model is implemented in Python 3.11 using Mesa 3.x for agent scheduling, NetworkX for social-network construction, and NumPy / Matplotlib for numerical computation and visualisation. All simulation runs are seeded via `numpy.random.seed(run_index)`, ensuring bit-for-bit reproducibility given the same Python and library versions. Results are serialised to JSON after each (α, run) cell; the plotting pipeline reads these files without re-running the simulation, so figures can be regenerated without recomputation.

**Table S9. Software dependencies.**

| Package | Version |
|---|---|
| Python | 3.11 |
| Mesa | ≥ 3.0 |
| NetworkX | ≥ 3.0 |
| NumPy | ≥ 1.24 |
| Matplotlib | ≥ 3.7 |
| SciPy | ≥ 1.10 |

The full codebase, including the parameter files and post-processing scripts referenced here, is available at the repository listed on the title page. Raw JSON result files for the main sweep (N = 20) require approximately 150 MB of disk space and are archived separately.
