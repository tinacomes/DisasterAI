# Parameter Calibration Strategy for DisasterAI

## Critical Finding: D and delta Are NOT USED

**The D and delta parameters are defined but never used in the actual rejection logic!**

```python
# Line 124-125: DEFINED
self.D = 2.0 if agent_type == "exploitative" else 4
self.delta = 3.5 if agent_type == "exploitative" else 1.2

# Lines 1179-1196: ACTUAL REJECTION LOGIC (uses hardcoded values)
level_diff = abs(reported_level - prior_level)
if level_diff >= 3:
    rejection_prob = 0.9 * prior_confidence  # Hardcoded!
elif level_diff >= 2:
    rejection_prob = 0.7 * prior_confidence  # Hardcoded!
elif level_diff >= 1:
    rejection_prob = 0.3 * prior_confidence  # Hardcoded!
```

**Recommendation**: Either (a) remove D/delta, or (b) integrate them into the rejection logic.

---

## Complete Parameter Inventory

### Group 1: ENVIRONMENT (5 params)
| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| `width`, `height` | 30, 30 | 10-50 | Grid dimensions |
| `share_of_disaster` | 0.15 | 0.05-0.30 | Fraction of grid affected |
| `disaster_dynamics` | 2 | 0-3 | Evolution speed (0=static) |
| `shock_probability` | 0.1 | 0-0.3 | Random shock frequency |
| `shock_magnitude` | 2 | 1-3 | Shock intensity |

### Group 2: POPULATION (4 params)
| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| `number_of_humans` | 100 | 50-200 | Total agents |
| `share_exploitative` | 0.5 | 0.2-0.8 | Exploiter ratio |
| `num_ai` | 5 | 1-10 | AI agents (hardcoded) |
| `share_confirming` | 0.7 | 0.5-0.9 | Social network density |

### Group 3: AI BEHAVIOR (2 params)
| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| `ai_alignment_level` | 0.3 | 0.0-1.0 | **KEY**: 0=truthful, 1=confirming |
| `low_trust_amplification_factor` | 0.3 | 0-1.0 | Amplify alignment when trust is low |

### Group 4: TRUST DYNAMICS (4 params)
| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| `initial_trust` | 0.3 | 0.1-0.5 | Starting human trust |
| `initial_ai_trust` | 0.25 | 0.1-0.5 | Starting AI trust |
| `exploit_trust_lr` | 0.03 | 0.01-0.1 | Exploiter trust learning rate |
| `explor_trust_lr` | 0.05 | 0.02-0.15 | Explorer trust learning rate |

### Group 5: Q-LEARNING (4 params)
| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| `learning_rate` | 0.15 | 0.05-0.3 | Q-value update rate |
| `epsilon` | 0.3 | 0.1-0.5 | Exploration probability |
| `exploit_friend_bias` | 0.1 | 0-0.3 | Exploiter bias toward friends |
| `exploit_self_bias` | 0.1 | 0-0.3 | Exploiter bias toward self |

### Group 6: BELIEF DYNAMICS (3 params - HARDCODED in agent)
| Parameter | Current Value | Purpose |
|-----------|---------------|---------|
| `D` | 2.0/4.0 | **UNUSED** - intended acceptance threshold |
| `delta` | 3.5/1.2 | **UNUSED** - intended sensitivity |
| `belief_learning_rate` | 0.4/0.9 | How much beliefs shift on acceptance |

### Group 7: RUMOR MECHANISM (5 params)
| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| `rumor_probability` | 0.3 | 0-1.0 | Chance component gets rumor |
| `rumor_intensity` | 1.0 | 0.5-2.0 | Rumor strength |
| `rumor_confidence` | 0.6 | 0.3-0.9 | Initial rumor confidence |
| `rumor_radius_factor` | 0.5 | 0.3-1.0 | Rumor size relative to disaster |
| `min_rumor_separation_factor` | 0.5 | 0.3-0.8 | Min distance from real epicenter |

### Group 8: UNUSED/UNCLEAR (3 params)
| Parameter | Status |
|-----------|--------|
| `trust_update_mode` | Appears unused |
| `exploitative_correction_factor` | Appears unused |
| `lambda_parameter` | Appears unused |

---

## Calibration Strategy: Hierarchical Sensitivity Analysis

### Phase 1: Fix Structural Issues First

Before calibration, fix the D/delta issue. Integrate them properly:

```python
# PROPOSED FIX for lines 1179-1196:
level_diff = abs(reported_level - prior_level)
if self.agent_type == "exploitative" and prior_confidence > 0.4:
    # Use D as threshold, delta as sensitivity
    if level_diff >= self.D + 1:
        rejection_prob = 0.9 * prior_confidence
    elif level_diff >= self.D:
        rejection_prob = (self.delta / 5.0) * prior_confidence
    elif level_diff >= self.D - 1:
        rejection_prob = (self.delta / 10.0) * prior_confidence
    else:
        rejection_prob = 0.0
```

### Phase 2: Parameter Grouping for Calibration

**Tier 1 - Research Questions (vary these)**:
- `ai_alignment_level` - PRIMARY manipulation
- `share_exploitative` - Agent composition

**Tier 2 - Learning Dynamics (calibrate first)**:
- `learning_rate`, `epsilon` - Q-learning core
- `exploit_trust_lr`, `explor_trust_lr` - Trust adaptation
- `D`, `delta` (once fixed) - Belief acceptance

**Tier 3 - Environment (set and hold)**:
- `share_of_disaster`, `disaster_dynamics`
- Grid size, agent count

**Tier 4 - Initial Conditions (sensitivity check)**:
- `initial_trust`, `initial_ai_trust`
- Rumor parameters

### Phase 3: Calibration Protocol

#### Step 1: Baseline Behavior Validation
Run with extreme settings to verify mechanisms work:

```python
# Test 1: Pure exploitation (should create echo chambers)
test_extreme_exploit = {
    'share_exploitative': 0.9,
    'ai_alignment_level': 0.9,  # Confirming AI
    'epsilon': 0.1,  # Low exploration
}
# Expected: High SECI, low belief accuracy

# Test 2: Pure exploration (should break echo chambers)
test_extreme_explore = {
    'share_exploitative': 0.1,
    'ai_alignment_level': 0.1,  # Truthful AI
    'epsilon': 0.5,  # High exploration
}
# Expected: Low SECI, high belief accuracy

# Test 3: Q-learning actually learns
# Compare Q-values at t=0 vs t=150 - should diverge based on experience
```

#### Step 2: One-At-A-Time (OAT) Sensitivity
For each Tier 2 parameter, sweep while holding others at baseline:

```python
# Example: Learning rate sensitivity
learning_rate_sweep = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
for lr in learning_rate_sweep:
    run_simulation(learning_rate=lr, **baseline_params)
    record: Q_convergence_speed, final_SECI, final_MAE
```

**Key metrics to track**:
- Q-value convergence (should stabilize by tick ~100)
- SECI trajectory (should diverge by agent type)
- MAE trajectory (should improve over time for explorers)
- Trust evolution (should differentiate good vs bad sources)

#### Step 3: Interaction Effects (2-factor)
Test key interactions:

```python
# Interaction 1: learning_rate × epsilon
# Too fast learning + too much exploration = unstable
# Too slow learning + too little exploration = no learning

# Interaction 2: D × trust_lr (once D is fixed)
# High D (strict acceptance) needs faster trust learning
# Low D (loose acceptance) can have slower trust learning

# Interaction 3: ai_alignment × exploit_trust_lr
# Confirming AI + slow trust learning = persistent bad trust
# Truthful AI + fast trust learning = quick adaptation
```

#### Step 4: Robustness Check
Run 10+ replications with best parameters to check variance:

```python
results = []
for seed in range(20):
    random.seed(seed)
    result = run_simulation(**calibrated_params)
    results.append(result)

# Check: CV < 0.2 for key metrics
```

---

## Specific Calibration Recommendations

### Q-Learning Parameters

| Parameter | Recommended | Rationale |
|-----------|-------------|-----------|
| `learning_rate` | 0.1-0.15 | Standard Q-learning range; too high = instability |
| `epsilon` | 0.2-0.3 | Enough exploration to discover; not so much it's random |
| `exploit_friend_bias` | 0.05-0.1 | Small bias to reflect social preference |
| `exploit_self_bias` | 0.05-0.1 | Small bias for status quo |

**Validation check**: Plot Q-value evolution - should see:
- Initial period of exploration (first 30-50 ticks)
- Convergence to stable values (by tick 100)
- Differentiation between good and bad sources

### Trust Learning Rates

| Parameter | Recommended | Rationale |
|-----------|-------------|-----------|
| `exploit_trust_lr` | 0.01-0.02 | Slow to match exploiter resistance to change |
| `explor_trust_lr` | 0.03-0.05 | 2-3x faster for explorer adaptability |

**Validation check**: Trust in confirming AI should:
- Stay stable for exploiters (they like confirmation)
- Decrease over time for explorers (they detect inaccuracy)

### D and delta (Once Fixed)

| Parameter | Recommended | Rationale |
|-----------|-------------|-----------|
| `D` (exploiter) | 1.5-2.5 | Accept only close matches |
| `D` (explorer) | 3.5-4.5 | Accept wider range |
| `delta` (exploiter) | 3.0-4.0 | High sensitivity to rejection |
| `delta` (explorer) | 1.0-1.5 | Low sensitivity |

**Validation check**: Exploiters should reject 60-80% of conflicting info; explorers should reject 10-30%

---

## Proposed Calibration Experiment File

```python
"""
calibration_experiments.py
Systematic parameter calibration for DisasterAI
"""

import numpy as np
from DisasterAI_Model import DisasterModel
import itertools

# Baseline parameters
BASELINE = {
    'share_exploitative': 0.5,
    'share_of_disaster': 0.15,
    'initial_trust': 0.3,
    'initial_ai_trust': 0.25,
    'number_of_humans': 100,
    'share_confirming': 0.7,
    'disaster_dynamics': 2,
    'ai_alignment_level': 0.5,
    'learning_rate': 0.1,
    'epsilon': 0.3,
    'exploit_trust_lr': 0.02,
    'explor_trust_lr': 0.04,
    'ticks': 150,
}

def run_calibration_sweep(param_name, values, n_reps=5):
    """Sweep one parameter while holding others at baseline."""
    results = []
    for val in values:
        params = BASELINE.copy()
        params[param_name] = val

        rep_results = []
        for rep in range(n_reps):
            model = DisasterModel(**params)
            for _ in range(params['ticks']):
                model.step()

            # Extract metrics
            final_seci_exploit = model.seci_data[-1][1] if model.seci_data else 0
            final_seci_explor = model.seci_data[-1][2] if model.seci_data else 0
            final_mae_exploit = model.belief_error_data[-1][1] if model.belief_error_data else 0
            final_mae_explor = model.belief_error_data[-1][2] if model.belief_error_data else 0

            rep_results.append({
                'seci_exploit': final_seci_exploit,
                'seci_explor': final_seci_explor,
                'mae_exploit': final_mae_exploit,
                'mae_explor': final_mae_explor,
            })

        results.append({
            'param_value': val,
            'mean_seci_exploit': np.mean([r['seci_exploit'] for r in rep_results]),
            'std_seci_exploit': np.std([r['seci_exploit'] for r in rep_results]),
            'mean_mae_exploit': np.mean([r['mae_exploit'] for r in rep_results]),
            'mean_mae_explor': np.mean([r['mae_explor'] for r in rep_results]),
        })

    return results

# Calibration sweeps
SWEEPS = {
    'learning_rate': [0.05, 0.1, 0.15, 0.2, 0.25],
    'epsilon': [0.1, 0.2, 0.3, 0.4, 0.5],
    'exploit_trust_lr': [0.01, 0.02, 0.03, 0.05, 0.08],
    'explor_trust_lr': [0.02, 0.04, 0.06, 0.08, 0.1],
}

if __name__ == "__main__":
    for param, values in SWEEPS.items():
        print(f"\n=== Sweeping {param} ===")
        results = run_calibration_sweep(param, values, n_reps=3)
        for r in results:
            print(f"  {param}={r['param_value']:.2f}: "
                  f"SECI_exp={r['mean_seci_exploit']:.3f}±{r['std_seci_exploit']:.3f}, "
                  f"MAE_exp={r['mean_mae_exploit']:.3f}")
```

---

## Summary: Calibration Priority

1. **IMMEDIATE**: Fix D/delta - they're unused!
2. **HIGH**: Calibrate Q-learning (`learning_rate`, `epsilon`) - core mechanism
3. **MEDIUM**: Calibrate trust rates - differentiate agent types
4. **LOW**: Calibrate initial conditions - should be robust to these

**Total parameters**: ~25
**Active parameters**: ~18 (after removing unused)
**Research variables**: 2 (`ai_alignment_level`, `share_exploitative`)
**Parameters needing calibration**: ~8

---

## Recommended Experimental Design

### Full Factorial on Research Variables
```
ai_alignment_level: [0.1, 0.5, 0.9]  # 3 levels
share_exploitative: [0.3, 0.5, 0.7]  # 3 levels
= 9 conditions × 10 replications = 90 runs
```

### Hold Fixed (Calibrated Values)
```
learning_rate: 0.1
epsilon: 0.25
exploit_trust_lr: 0.02
explor_trust_lr: 0.04
D_exploit: 2.0 (once fixed)
D_explore: 4.0 (once fixed)
```

This reduces the "awful lot of parameterisation" to a manageable set!
