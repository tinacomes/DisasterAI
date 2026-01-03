# DisasterAI Experimental Design Documentation

**Version:** 2.0
**Date:** January 2026
**Status:** Active Research Framework

---

## Table of Contents

1. [Overview](#overview)
2. [Core Design Principles](#core-design-principles)
3. [Temporal vs Snapshot Visualization](#temporal-vs-snapshot-visualization)
4. [Experiment A: Dual-Timeline Feedback](#experiment-a-dual-timeline-feedback)
5. [Experiment B: Filter Bubble Dynamics](#experiment-b-filter-bubble-dynamics)
6. [Model Parameters](#model-parameters)
7. [Metrics Reference](#metrics-reference)
8. [Data Collection Protocol](#data-collection-protocol)
9. [Analysis Guidelines](#analysis-guidelines)

---

## Overview

### Research Context

The DisasterAI project investigates how AI alignment affects trust dynamics, information accuracy, and filter bubble formation in disaster response scenarios. Our model simulates human agents with Q-learning capabilities interacting with AI sources of varying alignment levels.

### Key Research Questions

1. **Learning Mechanisms**: How do dual-timeline feedback mechanisms (fast info quality + slow relief outcome) shape agent learning?
2. **Filter Bubbles**: Can AI alignment create, amplify, or break social echo chambers?
3. **Trust Dynamics**: How does trust evolve differently for exploratory vs exploitative agents?
4. **Information Quality**: What is the relationship between AI alignment, trust, and belief accuracy over time?

---

## Core Design Principles

### 1. Temporal First

**ALL primary analyses must be temporal, not snapshot-based.**

- ✓ Track metrics at every simulation tick
- ✓ Visualize evolution over time (line plots)
- ✓ Analyze trajectories, not just endpoints
- ✗ Avoid final-state-only comparisons (except as supplements)

**Rationale**: Learning, trust, and filter bubbles are dynamic processes. Their trajectories reveal mechanisms that final states obscure.

### 2. Multiple Agent Types

**Track exploratory and exploitative agents separately.**

- **Exploratory**: Higher sensing radius, more exploration, faster trust learning
- **Exploitative**: Lower sensing radius, exploit known sources, slower trust updates

**Rationale**: Agent types respond differently to AI alignment due to their information-gathering strategies.

### 3. Controlled Comparison

**Use well-defined experimental conditions with clear hypotheses.**

- Control conditions (no AI when appropriate)
- Parametric variation (alignment levels: 0.1, 0.5, 0.9)
- Fixed random seeds for reproducibility
- Sufficient run length (150-200 ticks) for dynamics to emerge

### 4. Multi-Scale Metrics

**Measure individual, network, and system-level outcomes.**

- **Individual**: Q-values, trust, belief accuracy per agent
- **Network**: SECI (social echo chamber), friendship patterns
- **System**: AECI (AI reliance), overall relief effectiveness

---

## Temporal vs Snapshot Visualization

### What Makes a Visualization "Temporal"?

#### ✓ Temporal Visualizations (REQUIRED)

1. **Time-series line plots**: Metric vs tick
   - Example: Q-value evolution over 150 ticks
   - Shows: Learning trajectory, convergence, oscillations

2. **Event timelines**: Scatter plots with time on x-axis
   - Example: When info vs relief feedback occurs
   - Shows: Temporal patterns in feedback mechanisms

3. **Evolution comparisons**: Multiple conditions on same time axis
   - Example: SECI trajectories for different AI alignments
   - Shows: Divergence points, relative speeds of change

4. **Change-from-baseline plots**: Δ metric over time
   - Example: SECI change from initial 10-tick average
   - Shows: Direction and magnitude of temporal shifts

#### ✗ Snapshot Visualizations (Use Sparingly)

1. **Final-state bar charts**: Only final tick values
   - Use case: Summary comparison after temporal analysis
   - Limitation: Hides how agents got there

2. **Aggregate statistics**: Mean across entire run
   - Use case: Hypothesis testing supplements
   - Limitation: Obscures temporal dynamics

### Our Approach

**Primary (80% of plots)**: Temporal evolution
**Secondary (20% of plots)**: Final-state summaries for hypothesis testing

---

## Experiment A: Dual-Timeline Feedback

### Research Question

**Does the dual-timeline feedback mechanism correctly guide agent learning?**

Specifically:
- Do exploratory agents decrease Q(AI) with confirming AI due to fast info quality feedback?
- Do exploitative agents change more slowly due to reliance on delayed relief feedback?

### Experimental Design

#### Conditions

| Condition | AI Alignment | Description | Expected Q(AI) Trajectory |
|-----------|--------------|-------------|---------------------------|
| High      | 0.9          | Confirming AI | Decrease for exploratory |
| Low       | 0.1          | Truthful AI | Increase for exploratory |

#### Parameters

```python
{
    'share_exploitative': 0.5,      # 50/50 agent mix
    'share_of_disaster': 0.15,      # 15% affected area
    'initial_trust': 0.3,           # Low initial trust (Fix 1)
    'initial_ai_trust': 0.25,       # Skeptical starting point
    'number_of_humans': 100,
    'ticks': 150,                   # Sufficient for learning
    'learning_rate': 0.1,           # Q-learning rate
    'epsilon': 0.3,                 # Exploration rate
    'exploit_trust_lr': 0.015,      # Slow trust learning (exploitative)
    'explor_trust_lr': 0.03,        # Faster trust learning (exploratory)
}
```

#### Temporal Metrics (Collected Every Tick)

1. **Q-value Evolution**
   - `Q(A_0)`: Expected value of querying AI
   - `Q(human)`: Expected value of querying humans
   - `Q(self_action)`: Expected value of acting on own beliefs
   - **Why temporal?** Learning curves reveal convergence speed, oscillations, and differential response to feedback

2. **Trust Evolution**
   - Trust in AI sources (A_0, A_1, ...)
   - Trust in friends vs non-friends
   - **Why temporal?** Trust building shows when agents "figure out" AI reliability

3. **Feedback Event Timeline**
   - When info quality feedback occurs (tick of event)
   - When relief outcome feedback occurs (tick of event)
   - **Why temporal?** Reveals dual-timeline mechanism—fast vs slow feedback

4. **AI Usage Patterns**
   - Proportion of queries directed to AI over time
   - **Why temporal?** Shows behavioral shift as learning proceeds

#### Hypotheses

**H1**: Exploratory agents with confirming AI (0.9) will show **decreasing Q(A_0) over time** due to frequent negative info quality feedback.

**H2**: Exploratory agents with truthful AI (0.1) will show **increasing Q(A_0) over time** due to frequent positive info quality feedback.

**H3**: Exploitative agents will show **slower Q-value changes** than exploratory agents due to less frequent info quality feedback.

**H4**: Info quality feedback occurs **3-5 ticks after query**, while relief feedback occurs **15-25 ticks after action**.

#### Expected Temporal Patterns

```
Tick 0-30: Initial exploration phase
  - High variance in Q-values
  - Trust values near initial (0.25-0.3)
  - Frequent info quality feedback for exploratory

Tick 30-90: Learning phase
  - Q-values diverge based on AI alignment
  - Trust increases/decreases based on feedback
  - Exploratory Q(AI) drops with confirming AI
  - Exploratory Q(AI) rises with truthful AI

Tick 90-150: Convergence phase
  - Q-values stabilize (but not necessarily to zero!)
  - Trust plateaus
  - Behavior patterns solidify
```

### Visualization Requirements

**Must include (all temporal)**:

1. Q-value evolution line plots (separate for each agent type)
2. Trust evolution comparison (all conditions)
3. Feedback event timeline (scatter plot showing when feedback fires)
4. Feedback frequency bar chart (total counts as summary)

**Optional supplements**:

5. Final Q-value comparison (bar chart)
6. Summary statistics table

---

## Experiment B: Filter Bubble Dynamics

### Research Questions

1. Does AI alignment **create** filter bubbles where none existed?
2. Does AI alignment **amplify** existing social filter bubbles?
3. Can truthful AI **break** filter bubbles?

### Experimental Design

#### Conditions

| Condition        | AI Alignment | Purpose |
|------------------|--------------|---------|
| Control          | None         | Baseline social dynamics without AI |
| Truthful AI      | 0.1          | Test bubble-breaking hypothesis |
| Mixed AI         | 0.5          | Intermediate case |
| Confirming AI    | 0.9          | Test bubble-amplification hypothesis |

#### Parameters

```python
{
    'share_exploitative': 0.5,
    'share_of_disaster': 0.15,
    'initial_trust': 0.3,
    'initial_ai_trust': 0.25,
    'number_of_humans': 100,
    'share_confirming': 0.7,        # 70% of social network is confirming
    'disaster_dynamics': 2,
    'width': 30,
    'height': 30,
    'ticks': 200,                   # Longer for filter bubble evolution
    'learning_rate': 0.1,
    'epsilon': 0.3,
    'exploit_trust_lr': 0.015,
    'explor_trust_lr': 0.03,
}
```

#### Temporal Metrics

1. **SECI (Social Echo Chamber Index)**
   - Range: -1 (strong echo chamber) to +1 (anti-echo chamber)
   - 0 = friends as diverse as global population
   - **Collected**: Every tick
   - **Why temporal?** Shows filter bubble formation/dissolution trajectory

2. **AECI (AI Echo Chamber Index)**
   - Range: 0 (only queries humans) to 1 (only queries AI)
   - Normalized to [-1, +1] for comparison: 2*(AECI - 0.5)
   - **Collected**: Every tick
   - **Why temporal?** Tracks increasing/decreasing AI reliance

3. **Belief Accuracy (MAE)**
   - Mean Absolute Error of beliefs vs ground truth
   - **Collected**: Every 10 ticks
   - **Why temporal?** Information quality consequences of filter bubbles

4. **AI Usage Rate**
   - Proportion of queries to AI vs humans
   - **Collected**: Every 10 ticks
   - **Why temporal?** Behavioral shift toward/away from AI

#### Hypotheses

**H1 (Amplification)**: Confirming AI (0.9) **amplifies** social filter bubbles
- Prediction: SECI becomes **more negative over time** compared to control
- Mechanism: AI reinforces existing biases → reduced belief diversity

**H2 (Breaking)**: Truthful AI (0.1) **breaks** social filter bubbles
- Prediction: SECI becomes **less negative over time** compared to control
- Mechanism: AI provides diverse, accurate info → increased belief diversity

**H3 (AI-Social Interaction)**: High AECI + confirming AI creates **strongest filter bubbles**
- Prediction: Most negative SECI when AECI is high AND alignment is 0.9
- Mechanism: Heavy AI reliance + confirmation bias = echo chamber

**H4 (Agent Type)**: Exploratory agents show **weaker filter bubble effects**
- Prediction: Smaller SECI range across conditions for exploratory vs exploitative
- Mechanism: More sensing → less reliance on filtered social/AI info

#### Expected Temporal Patterns

```
Tick 0-50: Baseline period
  - All conditions show similar SECI (social structure dominant)
  - AECI near 0 (agents haven't learned to use AI much)
  - Belief accuracy similar across conditions

Tick 50-120: Divergence period
  - SECI trajectories diverge based on AI alignment
  - Confirming AI: SECI drops (stronger bubbles)
  - Truthful AI: SECI rises (weaker bubbles)
  - AECI increases for all AI conditions

Tick 120-200: Stabilization period
  - SECI differences plateau
  - AECI reaches equilibrium
  - Belief accuracy reflects filter bubble strength
```

### Visualization Requirements

**Must include (all temporal)**:

1. SECI evolution over time (separate plots for exploitative/exploratory)
2. AECI evolution over time (both agent types on same plot)
3. Belief accuracy evolution (MAE over time)
4. SECI change from baseline (Δ SECI trajectory)
5. Normalized AECI vs SECI comparison (both on -1 to +1 scale)

**Optional supplements**:

6. Final SECI comparison (bar chart)
7. Hypothesis testing summary (text table)

---

## Model Parameters

### Fixed Parameters (Across All Experiments)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `width` | 30 | Sufficient space for spatial dynamics |
| `height` | 30 | Balanced with agent count |
| `number_of_humans` | 100 | Large enough for network effects, small enough to track |
| `disaster_dynamics` | 2 | Medium disaster evolution speed |
| `learning_rate` | 0.1 | Standard Q-learning rate |
| `epsilon` | 0.3 | 30% exploration maintains diversity |

### Trust Parameters (Post-Fix 1 & 2)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `initial_trust` | 0.3 | Low enough to prevent blind faith (Fix 1) |
| `initial_ai_trust` | 0.25 | Skeptical starting point (Fix 1) |
| `exploit_trust_lr` | 0.015 | Slow trust building for exploiters (Fix 2) |
| `explor_trust_lr` | 0.03 | Faster but still cautious for explorers (Fix 2) |

### Varied Parameters

| Experiment | Parameter | Values | Purpose |
|------------|-----------|--------|---------|
| A | `ai_alignment_level` | 0.1, 0.9 | Test dual-timeline feedback |
| A | `ticks` | 150 | Sufficient for learning convergence |
| B | `ai_alignment_level` | None, 0.1, 0.5, 0.9 | Filter bubble continuum |
| B | `ticks` | 200 | Longer for social dynamics |

---

## Metrics Reference

### Q-Values

**Definition**: Expected cumulative reward for choosing a source mode

```
Q(mode) ← Q(mode) + α[R + γ·max(Q') - Q(mode)]
```

**Modes**:
- `A_k`: Query AI agent k
- `human`: Query human contacts
- `self_action`: Act on current beliefs

**Interpretation**:
- Positive Q-value: Agent expects this mode to be beneficial
- Negative Q-value: Agent expects this mode to be detrimental
- Higher Q-value → more likely to choose this mode (ε-greedy)

**Temporal dynamics**:
- Initially: All near 0
- Learning: Diverge based on feedback
- Convergence: Stabilize (but may not reach 0)

### Trust

**Definition**: Bayesian confidence in a source's reliability

**Range**: 0 (no trust) to 1 (complete trust)

**Update mechanism**:
- Positive feedback: `trust ← trust + lr·(1 - trust)`
- Negative feedback: `trust ← trust - lr·trust`

**Temporal dynamics**:
- Start: `initial_ai_trust` (0.25)
- Learning: Increases/decreases based on info quality
- Exploratory: Changes faster (higher `explor_trust_lr`)
- Exploitative: Changes slower (lower `exploit_trust_lr`)

### SECI (Social Echo Chamber Index)

**Definition**: Measures filter bubble strength in social network

```
SECI = (diversity_friends - diversity_global) / diversity_global
```

**Range**: -1 to +1
- **-1**: Perfect echo chamber (friends identical beliefs)
- **0**: No echo chamber effect (friends as diverse as population)
- **+1**: Anti-echo chamber (friends more diverse than population)

**Interpretation**:
- More negative → stronger filter bubble
- Movement toward 0 → bubble breaking
- Movement away from 0 (negative) → bubble amplification

**Temporal dynamics**:
- Early: Reflects initial social network structure
- Mid: AI influence begins affecting belief diversity
- Late: Equilibrium between social and AI effects

### AECI (AI Echo Chamber Index)

**Definition**: Proportion of information queries directed to AI

```
AECI = AI_queries / (AI_queries + human_queries)
```

**Range**: 0 to 1
- **0**: Only queries humans
- **1**: Only queries AI
- **0.5**: Equal human/AI usage

**Normalized to SECI scale**: `2*(AECI - 0.5)` → range [-1, +1]

**Interpretation**:
- High AECI: Heavy AI reliance
- Low AECI: Prefers human sources
- Increasing AECI: Learning to trust AI

**Temporal dynamics**:
- Early: Near 0 (agents haven't learned to use AI)
- Mid: Increases as agents evaluate AI quality
- Late: Reflects learned AI trust

### Belief Accuracy (MAE)

**Definition**: Mean Absolute Error between beliefs and ground truth

```
MAE = (1/N) Σ |belief_level - true_level|
```

**Range**: 0 (perfect accuracy) to 5 (maximum error)

**Interpretation**:
- Lower MAE → more accurate beliefs
- Higher MAE → less accurate (filter bubble effect)

**Temporal dynamics**:
- Early: High variance (limited information)
- Mid: Decreases as agents gather info
- Late: Reflects information quality of sources used

---

## Data Collection Protocol

### Tick-by-Tick Collection

**Priority 1 (Every Tick)**:
- Q-values for sample agents (one exploratory, one exploitative)
- Trust values for sample agents
- SECI for both agent types
- AECI for both agent types

**Priority 2 (Every 10 Ticks)**:
- Belief accuracy (MAE) for all agents
- AI usage rate
- Network statistics

**Priority 3 (Final Tick Only)**:
- Full agent state for post-hoc analysis
- Model snapshot for reproducibility

### Sample Agent Selection

For Q-value and trust tracking:
- Select **first encountered** exploratory agent
- Select **first encountered** exploitative agent
- Track same agents throughout run (no re-sampling)

**Rationale**: Consistent tracking shows individual learning trajectories

### Data Storage Format

```python
# Temporal data structure
{
    'q_values': {
        'exploratory': {
            'A_0': [tick_0_value, tick_1_value, ...],  # 150-200 values
            'human': [...],
            'self_action': [...]
        },
        'exploitative': {...}
    },
    'trust': {
        'exploratory': [tick_0_value, ...],
        'exploitative': [...]
    },
    'seci': {
        'exploit': [tick_0_value, ...],
        'explor': [...]
    },
    'feedback_timeline': {
        'info': [(tick, agent_type), ...],     # Temporal events
        'relief': [(tick, agent_type), ...]
    }
}
```

---

## Analysis Guidelines

### Primary Analysis: Temporal Trajectories

1. **Plot all metrics as time series first**
   - X-axis: Tick (0 to max_ticks)
   - Y-axis: Metric value
   - Separate lines for conditions/agent types

2. **Identify critical time windows**
   - When do trajectories diverge?
   - When do values stabilize?
   - Are there oscillations or monotonic changes?

3. **Compare rates of change**
   - Which condition changes fastest?
   - Do exploratory vs exploitative show different speeds?

### Secondary Analysis: Statistical Tests

Only after temporal analysis, test:

1. **Final-state differences** (t-tests, ANOVA)
   - Use final 20 ticks (averaged) for stability
   - Compare conditions

2. **Trajectory differences** (mixed-effects models)
   - Time × Condition interaction
   - Reveals divergence points

3. **Hypothesis testing**
   - Directional tests based on predictions
   - Effect sizes matter more than p-values

### Reporting Standards

**Every result must include**:

1. ✓ Temporal visualization (primary evidence)
2. ✓ Description of trajectory pattern
3. ✓ Comparison to hypotheses
4. ✓ Statistical test (if claiming difference)

**Avoid**:
- ✗ Reporting only final values
- ✗ Aggregate statistics without temporal context
- ✗ Claims about "learning" without showing learning curves

---

## Validation Checklist

Before analyzing results, confirm:

- [ ] All primary plots show time on x-axis
- [ ] Data collected at every tick (or documented sampling rate)
- [ ] Same agents tracked throughout run (no re-sampling)
- [ ] Hypotheses stated before looking at results
- [ ] Control conditions included where appropriate
- [ ] Agent type differences examined
- [ ] Temporal patterns described, not just final states

---

## References

### Related Documents

- `FIXES_SUMMARY.md`: Q-learning bug fixes (Issues #0-#3)
- `ISSUES_ANALYSIS.md`: Detailed issue analysis
- `test_dual_feedback.py`: Experiment A implementation
- `test_filter_bubbles.py`: Experiment B implementation
- `DisasterAI_Model.py`: Main model code

### Key Model Features

1. **Dual-timeline feedback** (lines 3500-3700 in DisasterAI_Model.py)
   - Info quality feedback: 3-5 tick delay
   - Relief outcome feedback: 15-25 tick delay

2. **Q-learning updates** (lines 3300-3500)
   - Separate updates for info quality vs relief
   - Agent type differences in learning rates

3. **Trust dynamics** (lines 2800-3100)
   - Bayesian updating
   - Different learning rates for exploratory/exploitative

4. **SECI calculation** (lines 6800-7000)
   - Friend belief diversity vs global diversity
   - Tracked separately by agent type

---

## Version History

**v2.0 (January 2026)**
- Comprehensive temporal-first design
- Two main experiments (A: dual-timeline, B: filter bubbles)
- Post-fix parameters (Fix 1 & 2 incorporated)
- Explicit temporal vs snapshot distinction

**v1.0 (December 2025)**
- Initial experimental framework
- Basic Q-learning experiments
- Pre-fix parameters

---

## Contact & Contributions

For questions about experimental design:
- Review this document first
- Check related documents (FIXES_SUMMARY.md, etc.)
- Examine test implementation files

When proposing new experiments:
- Follow temporal-first principle
- Define clear hypotheses before running
- Include appropriate controls
- Document parameter choices

---

**END OF EXPERIMENTAL DESIGN DOCUMENTATION**
