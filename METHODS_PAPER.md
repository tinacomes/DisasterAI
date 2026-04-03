# Methods

## Simulation Model

We develop an agent-based model (ABM) of disaster response in which human agents form beliefs about local disaster severity, seek information from social contacts or an AI source, and dispatch relief tokens accordingly. ABMs are well suited to this setting because filter-bubble phenomena emerge from the interaction of individual heuristics and network structure rather than from aggregate equilibria (Epstein 2006; Axelrod 1997). Unlike equation-based or compartmental models that assume homogeneous mixing, the framework here supports heterogeneous agent types, explicit social networks, and a spatially explicit disaster environment, all of which are necessary to capture the near/far coverage gradient and degree-based periphery effects that are our secondary outcomes.

The environment is a 30 × 30 grid in which a disaster evolves over discrete ticks. Each cell carries a severity level between 0 (no impact) and 5 (critical), initialised as a Gaussian decay from a randomly placed epicentre. At every tick each cell drifts stochastically towards its baseline value and receives random shocks (probability 0.10, magnitude ±2 levels), producing a non-stationary information landscape that prevents agents from converging to a single correct belief and forces ongoing information-seeking throughout the simulation.

One hundred human agents are distributed on the grid (population density fixed at N = 100 to keep the network tractable while preserving meaningful community structure). Agents belong to one of two cognitive types, each occupying equal shares of the population by default. *Exploitative* agents are confirmation-seeking: they have a narrow acceptance window D = 2.0 and a steep acceptance-sensitivity parameter δ = 3.5, making them resistant to information that diverges substantially from their prior. *Exploratory* agents are accuracy-seeking: D = 4.0 and δ = 1.2 give them a wide, gradual acceptance curve that admits novel information. These two archetypes map onto documented individual differences in information processing under uncertainty (Hertwig & Engel 2016; March 1991) and have been used in prior disaster-response ABMs to represent the range between cautious, community-focused responders and risk-tolerant, discovery-oriented ones.

The social network is constructed to reflect type-homogeneous community structure: exploitative agents are wired predominantly within their own cluster, exploratory agents within theirs, with no enforced cross-type edges. This follows empirical evidence of homophily in disaster communication networks (Bruns et al. 2012; Sutton et al. 2008) and ensures that the Social Echo Chamber Index (see below) captures genuine community-level belief convergence rather than individual coincidences.

Five AI agents serve the community. Each AI agent senses 15 % of the grid per tick and responds to human queries with a report whose truth content is controlled by the alignment parameter α (see next section). Agents query sources using an ε-greedy Q-learning strategy (ε = 0.3, learning rate η = 0.10), selecting among querying a social contact, querying the AI, or acting on current beliefs. Exploitative agents receive delayed outcome feedback (15–25 ticks) when their relief tokens are processed, updating Q-values slowly; exploratory agents also receive faster information-quality feedback, producing a dual-timeline reward structure that has been shown to generate more realistic trust trajectories than single-signal models (Fu et al. 2023).

## AI Alignment and Belief Dynamics

The alignment parameter α ∈ [0, 1] governs the truthfulness of AI reports. A queried AI agent responds with:

$$r_\text{AI} = (1 - \alpha)\, t + \alpha\, b$$

where $t$ is the cell's true severity and $b$ is the querying agent's current belief. At α = 0 the AI reports ground truth; at α = 1 it perfectly confirms the agent's prior; intermediate values interpolate linearly. This formulation isolates the confirmation effect from other AI failure modes (latency, coverage, bias), enabling a clean sweep across the truth–confirmation spectrum.

Upon receiving a report from any source, an agent updates its belief via a Bayesian acceptance mechanism. The probability of accepting the report depends on its distance from the current belief:

$$P(\text{accept}) = \frac{D_\text{eff}^\delta}{d^\delta + D_\text{eff}^\delta},\quad d = |r - b|$$

where $D_\text{eff} = D \cdot (1 + 0.5\, T_\text{source})$ for social contacts (trust-widened) and $D_\text{eff} = D$ for AI. When a report is accepted, the posterior is precision-weighted:

$$b' = \frac{\pi_\text{prior}\, b + \pi_\text{source}\, r}{\pi_\text{prior} + \pi_\text{source}}$$

with $\pi \propto c/(1-c)$ (odds of current confidence $c$), scaled by a type-specific factor (1.5 for exploitative, 0.8 for exploratory) that makes exploitative agents more resistant to belief revision. This mechanism generalises the Deffuant bounded-confidence model (Deffuant et al. 2000) by replacing a binary accept/reject threshold with a smooth probability curve, and by coupling threshold width to the source's trust score, producing richer dynamics than threshold models while remaining analytically interpretable.

## Echo-Chamber and Accuracy Metrics

We define three primary outcomes. The **Social Echo Chamber Index (SECI)** captures the extent to which an agent's social neighbourhood is epistemically insular:

$$\text{SECI} = 1 - \frac{\text{Var}(\text{beliefs of friends})}{\text{Var}(\text{beliefs of all agents})}$$

SECI < 0 indicates that friends hold more homogeneous beliefs than the global population (echo chamber); SECI = 0 is the null; SECI > 0 indicates friends are more diverse than the population (anti-echo). Computed every five ticks and reported separately for each agent type, SECI is a network-normalised measure that does not conflate individual accuracy with community insularity. Alternative formulations based on opinion fragmentation (Bail et al. 2018) or information entropy (Sasahara et al. 2021) were considered; the variance ratio was preferred because it has a natural zero, is comparable across agent types, and maps directly onto the information-theoretic notion of surprise that motivates the Bayesian update above.

The **AI Echo Chamber Index (AECI)** measures the fraction of all accepted information updates that originate from AI sources during any five-tick window:

$$\text{AECI} = \frac{\text{accepted}_\text{AI}}{\text{accepted}_\text{AI} + \text{accepted}_\text{human}}$$

Unlike a simple query-count ratio, this tracks whether AI information is actually absorbed into beliefs, distinguishing reliance from mere querying. High AECI signals over-dependence on a potentially confirming source, regardless of how often human contacts are also consulted.

**Belief accuracy (MAE)** is the mean absolute error between agent beliefs and true severity, averaged over cells with non-zero severity and over both agent types, sampled every five ticks. MAE enters the analysis as an operational criterion: a Goldilocks alignment should not merely suppress echo chambers but also maintain adequate situational awareness for effective relief delivery.

To locate the optimal alignment level we construct two composite scores over the steady-state window (last 75 simulation ticks). Both use range normalisation across the alignment sweep so that each component contributes on a [0, 1] scale:

$$\text{total\_bubble} = |\text{SECI}|_\text{norm} + |\text{AECI}|_\text{norm}$$
$$\text{total\_score} = |\text{SECI}|_\text{norm} + |\text{AECI}|_\text{norm} + \text{MAE}_\text{norm}$$

The Goldilocks optimum α* is the alignment level that minimises total\_bubble; α*(+MAE) is the minimiser of total\_score. Reporting both criteria makes the trade-off between echo-chamber suppression and epistemic accuracy explicit.

## Experimental Design

The primary experiment sweeps α across eleven levels (0.0, 0.1, …, 1.0) with N = 20 independent replications per level, each using a different random seed. Replications differ in epicentre location, agent initialisation, and stochastic shock sequences, ensuring that results reflect systematic effects of alignment rather than single-run artefacts. Boxplots across replications are shown for all steady-state scalar outcomes.

A secondary experiment varies the *cognitive gap scalar* g ∈ {0.0, 0.5, 1.0, 1.5}, which scales the difference between exploitative and exploratory acceptance parameters from a shared midpoint (D\_mid = 3.0, δ\_mid = 2.35) while preserving the invariant D\_exploit < D\_explor and δ\_exploit > δ\_explor. At g = 0 both agent types are cognitively identical, providing a null condition for heterogeneity effects; at g = 1 the parameters recover the baseline calibration. This sweep tests whether the Goldilocks α* is robust to the degree of within-population cognitive diversity.

Spatial and network periphery analyses decompose outcomes by proximity to the disaster epicentre (nearest-quartile vs furthest-quartile cells) and by social-network degree (Q1 vs Q4 agents). These periphery gaps address whether alignment benefits are equitably distributed or concentrated among well-connected, centrally located actors — a key equity concern in humanitarian AI applications.
