# Trust Dynamics Analysis & Experimental Design

## Problem Statement

**Current Issue**: Trust in AI peaks suddenly for ALL agent types, regardless of AI alignment level or agent personality (exploratory vs exploitative).

**Why This is Wrong**:
- Exploratory agents should REDUCE trust in confirming AI (high alignment)
- Exploitative agents should REDUCE trust in truthful AI (low alignment)
- Trust dynamics should show clear divergence based on agent type × AI alignment interaction
- Final Q-values are not meaningful - we need to track temporal evolution

## Root Cause Analysis

### 1. Trust Initialization Problem
Looking at the code, trust is initialized at:
- `initial_trust`: 0.3 (generic trust)
- `initial_ai_trust`: 0.25 (AI-specific trust)

**But**: These values may be too low OR the learning rates too aggressive, causing rapid convergence to ceiling/floor.

### 2. Trust Update Mechanisms

There are TWO feedback pathways:

#### A. Relief Outcome Feedback (Long Timeline: 15-25 ticks)
```python
# DisasterAI_Model.py:1348-1356
trust_change = trust_learning_rate * (target_trust - old_trust)
new_trust = max(0.0, min(1.0, old_trust + trust_change))
```

- Learning rates: `exploit_trust_lr=0.015`, `explor_trust_lr=0.03`
- Target trust based on relief success: 0.8-1.0 if success, 0.1-0.3 if failure
- **Problem**: If relief succeeds even once, trust can jump from 0.25 → 0.4+ immediately

#### B. Info Quality Feedback (Short Timeline: 3-15 ticks)
```python
# DisasterAI_Model.py:443-449
trust_adjustment = info_learning_rate * accuracy_reward
new_trust = max(0.0, min(1.0, old_trust + trust_adjustment))
```

- Accuracy reward: +1.0 if accurate, -1.0 if inaccurate
- **Problem**: Single accurate prediction can boost trust significantly

### 3. Trust Decay
```python
# DisasterAI_Model.py:693
self.trust[source_id] = max(0, self.trust[source_id] - decay_rate)
# decay_rate = 0.01 (non-friends) or 0.002 (friends)
```

**Problem**: Decay is very slow compared to positive updates. Trust can spike and stay elevated.

### 4. The "Lucky Confirmation" Problem

Current scenario:
1. Agent queries confirming AI (high alignment)
2. AI confirms existing belief (even if wrong)
3. Agent sends relief based on that belief
4. Relief happens to succeed (lucky coincidence or belief was actually correct)
5. **Trust in AI skyrockets** due to relief success
6. Agent continues trusting confirming AI

This is the OPPOSITE of what should happen for exploratory agents!

## Why Final Q-Values Are Not Meaningful

Q-values are updated based on:
- Relief outcome (long delay, noisy)
- Info quality (variable delay, depends on sensing overlap)

**Problems**:
1. **Sparse updates**: If agent rarely uses a source, Q-value barely changes from initialization
2. **Path dependency**: Final Q-value depends heavily on early lucky/unlucky experiences
3. **Non-stationarity**: Disaster grid evolves, so "true" Q-value changes over time
4. **Exploration-exploitation tradeoff**: ε-greedy means agents may keep using bad sources

**Better metric**: Temporal evolution shows:
- When does trust peak?
- How quickly does trust change?
- Does trust plateau or oscillate?
- Are there clear divergence points between conditions?

## Proposed Experimental Design for Colab

### Experiment 1: Trust Dynamics Temporal Analysis

**Goal**: Map out complete temporal evolution of trust for all agent type × AI alignment combinations

**Design**:
```python
# Conditions (2×4 = 8 total)
agent_types = ['exploratory', 'exploitative']
ai_alignments = [None, 0.1, 0.5, 0.9]  # Control, Truthful, Mixed, Confirming

# Track every tick (not just final)
metrics_per_tick = {
    'ai_trust_mean': [],    # Average trust in AI
    'ai_trust_std': [],     # Variance across agents
    'ai_trust_q25': [],     # 25th percentile
    'ai_trust_q75': [],     # 75th percentile
    'ai_usage_rate': [],    # % of agents querying AI
    'trust_velocity': [],   # Rate of change (Δtrust/Δtick)
    'trust_acceleration': [] # Second derivative
}

# Critical time windows
early_phase = ticks[0:30]      # Initial learning
middle_phase = ticks[30:100]   # Consolidation
late_phase = ticks[100:200]    # Equilibrium?
```

**Hypotheses**:
- H1: Exploratory + Confirming AI shows early peak then decline
- H2: Exploratory + Truthful AI shows steady increase
- H3: Exploitative + Confirming AI shows sustained high trust
- H4: Exploitative + Truthful AI shows sustained low trust
- H5: Peak timing differs between agent types (exploratory peaks earlier)

**Visualization**:
- 2×2 grid: rows=agent type, cols=AI alignment level
- Each subplot: trust evolution with confidence bands
- Overlay: mark "peak tick" for each condition
- Add: trust velocity subplot (rate of change)

### Experiment 2: Feedback Mechanism Ablation

**Goal**: Isolate which feedback mechanism drives the sudden trust peaks

**Design**:
```python
# Test conditions (disable one mechanism at a time)
conditions = [
    'both_enabled',           # Baseline
    'relief_only',           # Disable info quality feedback
    'info_only',             # Disable relief outcome feedback
    'neither'                # Pure decay (negative control)
]

# For each condition, track:
- Trust evolution
- Q-value evolution
- Feedback event counts
- Time to first peak
```

**Expected Results**:
- If relief feedback causes peaks → 'relief_only' shows peaks
- If info feedback causes peaks → 'info_only' shows peaks
- If both needed → only 'both_enabled' shows peaks

### Experiment 3: Learning Rate Sensitivity

**Goal**: Test if learning rates are too aggressive

**Design**:
```python
# Current values (from Fix 2)
exploit_trust_lr_values = [0.005, 0.01, 0.015, 0.03]
explor_trust_lr_values = [0.01, 0.02, 0.03, 0.06]

# For high alignment (confirming AI) with exploratory agents
# Hypothesis: Lower LR prevents sudden peaks

# Track:
- Max trust achieved
- Tick at max trust
- Trust at end
- Variance over time (high variance = oscillation)
```

**Success Criteria**:
- Trust increases gradually (no sudden jumps)
- Final trust reflects cumulative evidence
- Exploratory agents still differentiate between alignments

### Experiment 4: Trust Decay Rebalancing

**Goal**: Test if decay should be more aggressive to counteract peaks

**Design**:
```python
decay_rates = {
    'current': {'non_friend': 0.01, 'friend': 0.002, 'ai': 0.005},
    'moderate': {'non_friend': 0.02, 'friend': 0.005, 'ai': 0.015},
    'aggressive': {'non_friend': 0.03, 'friend': 0.01, 'ai': 0.025}
}

# Hypothesis: Stronger decay prevents runaway trust
# But: Should not prevent learning good sources
```

### Experiment 5: Alternative Trust Update Rule

**Goal**: Test alternative update mechanisms that prevent spikes

**Design**:

#### Option A: Running Average (Anti-Recency)
```python
# Instead of single-update learning rate
# Use exponential moving average
trust_new = (1 - alpha) * trust_old + alpha * trust_target
# where alpha is MUCH smaller (0.01-0.05)
```

#### Option B: Evidence Accumulation
```python
# Track evidence count
evidence_count[source] += 1
evidence_positive[source] += (1 if reward > 0 else 0)

# Trust = proportion of positive evidence
trust[source] = evidence_positive[source] / evidence_count[source]
```

#### Option C: Bayesian Updating with Prior Strength
```python
# Stronger prior = more evidence needed to change trust
prior_strength = 10  # Equivalent to 10 observations

posterior_trust = (prior_strength * prior_trust + new_evidence) / (prior_strength + 1)
```

## Recommended Colab Notebook Structure

```python
# Section 1: Setup and Imports
import numpy as np
import matplotlib.pyplot as plt
from DisasterAI_Model import DisasterModel, HumanAgent
import pandas as pd
import seaborn as sns

# Section 2: Experiment 1 - Trust Dynamics
def run_trust_dynamics_experiment():
    """Map temporal evolution for all conditions"""
    # ... implementation ...
    return results_df

# Section 3: Experiment 2 - Feedback Ablation
def run_feedback_ablation_experiment():
    """Test which mechanism causes peaks"""
    # ... implementation ...
    return results_df

# Section 4: Experiment 3 - Learning Rate Sensitivity
def run_learning_rate_experiment():
    """Find optimal learning rates"""
    # ... implementation ...
    return results_df

# Section 5: Visualization Dashboard
def create_trust_dashboard(all_results):
    """Comprehensive visualization"""
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))

    # Row 1: Temporal evolution by agent type
    # Row 2: Peak timing analysis
    # Row 3: Trust velocity (rate of change)
    # Row 4: Feedback mechanism contribution

    return fig

# Section 6: Statistical Analysis
def analyze_trust_peaks(results_df):
    """Statistical tests for peak timing and magnitude"""
    # T-tests for peak differences
    # ANOVA for alignment effect
    # Interaction effects
    return stats_summary

# Section 7: Executive Summary
def generate_report(all_results, stats):
    """Auto-generate findings report"""
    # Summary of which hypotheses supported
    # Recommended parameter changes
    # Next steps
    return report_text
```

## Immediate Next Steps

### Quick Diagnostic (Run this FIRST)
```python
# Single run with diagnostic output
model = DisasterModel(
    ai_alignment_level=0.9,  # Confirming AI
    share_exploitative=0.5,
    initial_ai_trust=0.25,
    explor_trust_lr=0.03,
    ticks=200
)

# Track EVERY agent, EVERY tick
trust_matrix = np.zeros((num_agents, num_ticks))

for tick in range(200):
    model.step()

    # Record all agents' AI trust
    for i, agent in enumerate(model.agent_list):
        if isinstance(agent, HumanAgent):
            trust_matrix[i, tick] = agent.trust.get('A_0', 0.25)

# Find peaks
peak_ticks = np.argmax(trust_matrix, axis=1)
peak_values = np.max(trust_matrix, axis=1)

print(f"Mean peak tick: {peak_ticks.mean():.1f}")
print(f"Mean peak value: {peak_values.mean():.3f}")
print(f"Agents peaking before tick 30: {np.sum(peak_ticks < 30)}/{len(peak_ticks)}")
```

**If you see**:
- Mean peak < tick 30 → **Early peak problem confirmed**
- Peak value > 0.8 → **Ceiling effect confirmed**
- All agents peak similar time → **Systemic issue, not variance**

### Then Run Full Experiment 1
Focus on temporal dynamics to understand:
- WHEN do peaks occur?
- HOW fast does trust increase?
- DOES it plateau or decline after peak?

## Key Metrics to Report

Instead of "final AI trust = 0.73":

Report:
```
Trust Dynamics Summary:
- Initial: 0.25 (tick 0)
- First peak: 0.82 (tick 23)  ← PROBLEMATIC!
- Decline phase: tick 23-45
- Equilibrium: 0.51 (tick 100-200)
- Velocity max: +0.12/tick (tick 15-20)
- Half-life: 18 ticks (time to reach 50% of final value)
```

This tells a STORY about how trust evolves, not just where it ends up.

## Code Changes Needed (If Experiment Confirms)

### Fix 3: Slower Trust Updates
```python
# Current
exploit_trust_lr = 0.015
explor_trust_lr = 0.03

# Proposed (divide by 3-5x)
exploit_trust_lr = 0.003
explor_trust_lr = 0.006
```

### Fix 4: Stronger Trust Decay
```python
# Current
decay_rate = 0.01 if candidate not in self.friends else 0.002

# Proposed (multiply by 2-3x)
decay_rate = 0.025 if candidate not in self.friends else 0.005
```

### Fix 5: Evidence-Based Trust Cap
```python
# Don't allow trust to exceed evidence strength
min_queries_for_high_trust = 10

if source_id in self.query_count:
    max_allowed_trust = min(1.0, 0.5 + self.query_count[source_id] / min_queries_for_high_trust)
    self.trust[source_id] = min(self.trust[source_id], max_allowed_trust)
```

## Expected Timeline

1. **Quick Diagnostic** (30 min): Confirm peak timing issue
2. **Experiment 1** (2-3 hours): Full temporal analysis
3. **Analysis** (1 hour): Interpret results, test hypotheses
4. **Experiments 2-3** (2-3 hours): Ablation and sensitivity if needed
5. **Parameter tuning** (1-2 hours): Find optimal values
6. **Validation run** (1 hour): Confirm fixes work

**Total**: 1 day of focused work

## Success Criteria

✅ **Good Trust Dynamics**:
- Gradual increase/decrease (no sudden jumps)
- Clear divergence between agent types by tick 50
- Exploratory + Confirming: trust DECLINES or stays low
- Exploratory + Truthful: trust INCREASES steadily
- Exploitative: opposite pattern
- Final values reflect cumulative evidence, not lucky early events

❌ **Bad Trust Dynamics** (current state):
- Sudden spike in first 20-30 ticks
- All agent types converge to similar trust values
- High variance between runs (path dependency)
- Trust plateaus at ceiling (0.9+) or floor (0.1-)
