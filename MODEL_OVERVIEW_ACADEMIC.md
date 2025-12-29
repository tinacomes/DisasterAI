# DisasterAI Model: Academic Overview

## Model Architecture for Publication

### Abstract Description

The DisasterAI model is an agent-based model (ABM) implemented in Mesa that simulates information-seeking behavior and filter bubble formation in disaster response scenarios. The model features heterogeneous human agents with different behavioral strategies (exploitative vs. exploratory) and AI agents that provide information with varying degrees of alignment to human beliefs. Agents use Q-learning to select information sources and Bayesian updating to revise beliefs, creating emergent patterns of information segregation.

---

## 1. Model Components

### 1.1 Environment
- **Spatial Grid**: 30×30 discrete cells (configurable)
- **Disaster Distribution**: Gaussian decay from epicenter, levels L0-L5
- **Dynamics**: Stochastic hotspots (10% probability per tick) simulate evolving disaster
- **Duration**: 150 time steps (configurable)

### 1.2 Agent Types

#### Human Agents (N=30, configurable)
Two behavioral types with distinct characteristics:

**Exploitative Agents (50% of population)**
- Goal: Confirm existing beliefs, maintain social cohesion
- Strategy: Prefer friends and self-action
- Sensing radius: 2 cells (Moore neighborhood)
- Trust learning rate: 0.03 (slower)
- Belief learning rate: 0.4 (resistant to change)
- Source biases: +0.1 for friends, +0.1 for self-action

**Exploratory Agents (50% of population)**
- Goal: Maximize accuracy, explore uncertainty
- Strategy: Seek truthful information, explore uncertain areas
- Sensing radius: 3 cells (Moore neighborhood)
- Trust learning rate: 0.08 (faster)
- Belief learning rate: 0.9 (adaptive to new information)
- Source biases: -0.05 for self-action (anti-confirmation)

#### AI Agents (N=5, fixed)
- Knowledge coverage: 15% of grid per time step
- Memory persistence: 80% retention across steps
- Alignment parameter: Controls information manipulation (0=truthful, 1=fully aligned)
- Guessing capability: 75% probability to interpolate unknown cells

### 1.3 Social Network
- Structure: Multiple connected components (2-3 communities)
- Topology: Small-world properties with homophily
- Average degree: ~4-6 connections per agent
- Function: Defines friendship ties and information flow

### 1.4 Rumor Mechanism
- Probability: 30% per network component
- Intensity: Peak at approximately L4
- Radius: 50% of disaster radius
- Separation: Minimum 50% of disaster radius from true epicenter
- Propagation: Shared within network components

---

## 2. Core Mechanisms

### 2.1 Q-Learning for Source Selection

Agents maintain Q-values for each information source:
- **Sources**: Self-action, human network, AI agents (A₀...A₄)
- **Algorithm**: ε-greedy (ε=0.3)
  - Exploration (30%): Random source selection
  - Exploitation (70%): Select source with highest adjusted Q-value
- **Learning rate**: α = 0.15
- **Reward calculation**: Based on disaster relief accuracy (delayed by 2 ticks)

**Q-value Update**:
```
Q(s) ← Q(s) + α · [R - Q(s)]
```

Where:
- s = information source
- R = scaled reward ∈ [-1, 1]
- α = learning rate

**Reward Function**:
- L5 targeting: +5.0
- L4 targeting: +3.0
- L3 targeting: +1.5
- L2 targeting: 0.0
- L1 targeting: -1.0
- L0 targeting: -2.0

Agent-type specific weighting:
- Exploitative: 80% correctness ratio + 20% actual accuracy
- Exploratory: 80% actual accuracy + 20% correctness ratio

### 2.2 Bayesian Belief Updating

Agents update beliefs using precision-weighted Bayesian inference:

**Precision Calculation**:
```
π_prior = κ · C / (1 - C)
π_source = λ · T / (1 - T)
```

Where:
- C = confidence in current belief
- T = trust in information source
- κ = agent-type factor (1.8 for exploitative, 0.8 for exploratory)
- λ = source-type factor (1.0-12.0 depending on source and trust)

**Posterior Update**:
```
L_posterior = (π_prior · L_prior + π_source · L_reported) / (π_prior + π_source)
C_posterior = π_posterior / (1 + π_posterior)
```

Where π_posterior = π_prior + π_source

**Acceptance Threshold**: |L_posterior - L_prior| ≥ 1

### 2.3 Trust Dynamics

**Trust Initialization**:
- Human trust: 0.5 ± 0.05 (friends: +0.1 boost)
- AI trust: 0.5 ± 0.1

**Trust Update** (after reward processing):
```
T_new = T_old + β · (T_target - T_old)
```

Where:
- β = trust learning rate (agent-type dependent)
- T_target = (R + 1) / 2 (maps reward to [0,1])

**Trust Decay**:
- Friends: 0.0005/tick (exploitative), 0.0015/tick (exploratory)
- Non-friends: 0.002/tick (exploitative), 0.003/tick (exploratory)
- AI: Modulated by alignment level

### 2.4 AI Alignment Mechanism

AI agents manipulate reported information based on alignment level (α) and caller's beliefs:

**Alignment Factor**:
```
A_factor = α · (1 + 2C_caller) + α · λ_low · (1 - T_caller)
```

Where:
- α = alignment level ∈ [0, 1]
- C_caller = caller's confidence in their belief
- T_caller = caller's trust in AI
- λ_low = low-trust amplification factor (0.3)

**Adjusted Report**:
```
L_reported = L_sensed + A_factor · (L_caller - L_sensed)
```

**Effect**:
- α = 0: Pure truth (no adjustment)
- α = 0.3: Moderate alignment (baseline)
- α = 1.0: Full alignment (confirms human beliefs)

---

## 3. Agent Decision Cycle

Each agent executes the following cycle per time step:

1. **Sense Environment** (radius: 2-3 cells)
   - Observe local disaster levels with 8% noise probability
   - Update beliefs with distance-weighted blending

2. **Seek Information**
   - Determine interest point (exploitative: epicenter, exploratory: uncertain areas)
   - Select source via ε-greedy Q-learning with agent-type biases
   - Query source for information (radius: 2-3 cells)
   - Update beliefs via Bayesian mechanism

3. **Send Relief**
   - Target top 5 highest-belief cells (L≥3)
   - Scoring:
     - Exploitative: (L/5) · C^1.5
     - Exploratory: 0.7 · (L/5) + 0.3 · (1-C)
   - Queue rewards for evaluation (2-tick delay)

4. **Process Rewards**
   - Evaluate relief accuracy against ground truth
   - Update Q-values for used sources
   - Update trust in sources
   - Correct beliefs toward reality (70% weight)

5. **Maintenance**
   - Apply confidence decay (0.0003-0.0005/tick)
   - Apply trust decay (0.0005-0.003/tick)
   - Clean up tracking dictionaries (10-tick threshold)

---

## 4. Key Output Metrics

### 4.1 Social Echo Chamber Index (SECI)
Measures belief homogeneity within vs. between social network components:

```
SECI = (σ_between² - σ_within²) / σ_global²
```

Range: [-1, +1]
- SECI > 0: Social bubbles (within-group similarity)
- SECI < 0: Social diversity (between-group similarity)

### 4.2 AI Echo Chamber Index (AECI)
Measures belief variance among AI-reliant vs. all agents:

```
AECI = (σ_global² - σ_AI-reliant²) / σ_global²
```

Range: [-1, +1]
- AECI > 0: AI reduces diversity (echo chamber)
- AECI < 0: AI increases diversity (bubble breaking)

AI-reliant agents defined as: ≥2 AI calls AND ≥10% AI ratio

### 4.3 Belief Accuracy (MAE)
Mean Absolute Error between believed and actual disaster levels:

```
MAE = (1/N) Σ |L_believed - L_actual|
```

### 4.4 Specialization Rate
Percentage of agents relying on single source type:

```
SR = (N_AI-only + N_friend-only) / N_total
```

---

## 5. Experimental Parameters

### 5.1 Baseline Configuration
```python
{
    "share_exploitative": 0.5,        # Agent type ratio
    "share_of_disaster": 0.2,         # Grid coverage
    "initial_trust": 0.5,             # Human trust
    "initial_ai_trust": 0.5,          # AI trust
    "number_of_humans": 30,           # Population size
    "ai_alignment_level": 0.3,        # AI manipulation level
    "learning_rate": 0.15,            # Q-learning rate
    "epsilon": 0.3,                   # Exploration rate
    "ticks": 150,                     # Simulation duration
    "rumor_probability": 0.3          # Misinformation frequency
}
```

### 5.2 Experimental Manipulations

**Experiment A: Agent Type Ratio**
- Variable: `share_exploitative` ∈ {0.3, 0.5, 0.7}
- Question: How does agent heterogeneity affect filter bubbles?

**Experiment B: AI Alignment Tipping Points**
- Variable: `ai_alignment_level` ∈ [0.0, 1.0] (fine-grained)
- Question: At what alignment do exploratory agents switch from AI to social sources?

**Experiment C: Disaster Dynamics**
- Variables: `disaster_dynamics` ∈ {1,2,3}, `shock_magnitude` ∈ {1,2,3}
- Question: How does environmental volatility affect information strategies?

**Experiment D: Learning Parameters**
- Variables: `learning_rate` ∈ {0.03, 0.05, 0.07}, `epsilon` ∈ {0.2, 0.3}
- Question: How do learning speeds affect specialization?

---

## 6. Model Validation

### 6.1 Face Validity
- Exploitative agents show confirmation bias (prefer friends)
- Exploratory agents seek accuracy (prefer truthful AI)
- Filter bubbles emerge from decentralized decisions

### 6.2 Behavioral Validity
- Agent differentiation confirmed via acceptance patterns
- Q-learning convergence observed (~30-50 ticks)
- Trust dynamics follow reinforcement learning principles

### 6.3 Sensitivity Analysis
- Results robust to ±10% parameter variations
- Tipping points reproducible across runs (with random seed control)

---

## 7. Key Theoretical Contributions

1. **Q-Learning + Opinion Dynamics**: Novel integration of reinforcement learning with Bayesian belief updating

2. **AI Alignment as Information Manipulation**: Formalization of how AI systems may create or break filter bubbles based on alignment incentives

3. **Heterogeneous Agent Strategies**: Distinction between exploitative (social-confirmation) and exploratory (accuracy-seeking) agents grounded in behavioral economics

4. **Dynamic Filter Bubble Metrics**: SECI and AECI provide quantitative measures of echo chamber formation

---

## 8. Limitations and Future Work

### Current Limitations
- Simplified disaster dynamics (single Gaussian)
- Binary agent types (continuous trait distribution more realistic)
- Homogeneous AI agents (no AI specialization)
- Fixed social network (no dynamic relationship formation)

### Extensions
- Multi-issue belief spaces (beyond single disaster level)
- Adversarial AI agents (strategic misinformation)
- Network adaptation (dynamic friendship formation)
- Intervention strategies (platform design, regulation)

---

## 9. Reproducibility

### Code Availability
- Implementation: Mesa framework (Python)
- Repository: [GitHub URL]
- Version: Python 3.8+, Mesa 2.0+
- Random seeds: Documented for all experiments

### Computational Requirements
- Single run: ~2-5 minutes (standard laptop)
- Full experiment suite: ~2-4 hours (30 cores)
- Memory: <2GB per run

---

## References

Mesa Framework: https://mesa.readthedocs.io/
Agent-Based Modeling: Wilensky & Rand (2015)
Q-Learning: Sutton & Barto (2018)
Opinion Dynamics: Hegselmann & Krause (2002)
Filter Bubbles: Pariser (2011), Bakshy et al. (2015)

---

**Citation Format**:
[Author Name]. (2025). "Filter Bubble Dynamics in AI-Mediated Disaster Response: An Agent-Based Model." *Journal Name*. DOI: [to be assigned]

---

*Document Version: 1.0*
*Last Updated: 2025-12-29*
*Model Version: DisasterAI_Model.py (with bug fixes)*
