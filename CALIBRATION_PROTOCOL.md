# Calibration Protocol for DisasterAI

## Overview

This protocol provides systematic calibration for the memory-based belief system parameters. The goal is to find parameter values that produce:
1. **Differentiated behavior** between exploiters and explorers
2. **Realistic learning dynamics** (not too fast, not too slow)
3. **Meaningful echo chamber formation** under high-alignment AI
4. **Improved accuracy** under low-alignment AI

---

## Parameter Inventory

### Tier 1: Research Variables (DO NOT CALIBRATE - These are experimental manipulations)
| Parameter | Values | Purpose |
|-----------|--------|---------|
| `ai_alignment_level` | [0.1, 0.5, 0.9] | Primary IV: truthful vs confirming AI |
| `share_exploitative` | [0.3, 0.5, 0.7] | Secondary IV: population composition |

### Tier 2: Belief Acceptance (CALIBRATE FIRST - Core mechanism)
| Parameter | Current | Range | Agent Type |
|-----------|---------|-------|------------|
| `D` | 1.5 / 3.0 | [1.0-2.5] / [2.0-4.0] | Exploit / Explore |
| `delta` | 20 / 8 | [10-30] / [4-15] | Exploit / Explore |
| `memory_size` | 3 / 7 | [2-5] / [5-10] | Exploit / Explore |

### Tier 3: Learning Dynamics (CALIBRATE SECOND)
| Parameter | Current | Range | Purpose |
|-----------|---------|-------|---------|
| `learning_rate` | 0.15 | [0.05-0.25] | Q-value update speed |
| `epsilon` | 0.3 | [0.1-0.5] | Exploration vs exploitation |
| `exploit_trust_lr` | 0.03 | [0.01-0.08] | Exploiter trust adaptation |
| `explor_trust_lr` | 0.05 | [0.02-0.12] | Explorer trust adaptation |

### Tier 4: Source Precision Multipliers (CALIBRATE IF NEEDED)
| Parameter | Current | Range | Purpose |
|-----------|---------|-------|---------|
| `sensing_multiplier` | 1.5 / 0.7 | [1.2-2.0] / [0.5-0.9] | Sensed / Guessed |
| `type_multiplier` | 1.2 / 0.9 | [1.0-1.5] / [0.7-1.0] | Explorer / Exploiter source |
| `friend_multiplier` | 1.3 | [1.1-1.5] | Friend boost for exploiters |

---

## Validation Metrics

### Primary Metrics (Must show expected patterns)

1. **Acceptance Rate Differentiation**
   - Exploiters should accept 20-40% of conflicting info (diff ≥ 2)
   - Explorers should accept 60-80% of conflicting info (diff ≥ 2)
   - Ratio: Explorer acceptance / Exploiter acceptance > 2.0

2. **SECI Trajectory**
   - High-alignment AI: SECI should decrease (more echo chambers)
   - Low-alignment AI: SECI should stay stable or increase
   - Exploiters should have lower SECI than explorers

3. **Belief Accuracy (MAE)**
   - Explorers should have lower MAE than exploiters
   - Low-alignment AI should produce lower MAE than high-alignment
   - MAE should decrease over time (learning)

4. **Trust Differentiation**
   - With confirming AI: Exploiters should trust AI more than explorers
   - With truthful AI: Explorers should trust AI more than exploiters
   - Trust should converge to reflect actual source quality

### Secondary Metrics (Should show reasonable behavior)

5. **Q-Value Convergence**
   - Q-values should stabilize by tick 100
   - Coefficient of variation < 0.1 in final 50 ticks

6. **Memory Utilization**
   - Average memory fullness > 70% by tick 50
   - Memory should turn over (not stagnant)

---

## Calibration Protocol

### Phase 1: Baseline Validation (Sanity Checks)

Before calibration, verify the mechanism works as intended.

```python
# Test 1: Acceptance formula produces expected probabilities
# Exploiter at diff=2 should have P(accept) < 0.1
# Explorer at diff=2 should have P(accept) > 0.8

# Test 2: Memory fills and turns over
# After 20 ticks, most cells should have memory entries
# Memory should be at or near capacity

# Test 3: Extreme conditions produce extreme results
# All exploiters + high-alignment AI = strong echo chambers
# All explorers + low-alignment AI = high accuracy, low echo chambers
```

### Phase 2: D/δ Calibration (Most Important)

**Goal**: Find D/δ values that produce target acceptance rates.

**Method**: Sweep D and δ, measure acceptance rate at different level differences.

| Sweep | Exploiter D | Exploiter δ | Explorer D | Explorer δ |
|-------|-------------|-------------|------------|------------|
| 1 | [1.0, 1.5, 2.0, 2.5] | 20 | 3.0 | 8 |
| 2 | Best from 1 | [10, 15, 20, 25, 30] | 3.0 | 8 |
| 3 | Best | Best | [2.0, 2.5, 3.0, 3.5, 4.0] | 8 |
| 4 | Best | Best | Best | [4, 6, 8, 10, 12, 15] |

**Target**:
- Exploiter: 50% acceptance at diff=1, <5% at diff=2
- Explorer: 50% acceptance at diff=3, >80% at diff=2

### Phase 3: Memory Size Calibration

**Goal**: Find memory sizes that balance responsiveness with stability.

**Method**: Sweep memory_size, measure belief stability and adaptation speed.

| Agent Type | Sweep Values | Target Behavior |
|------------|--------------|-----------------|
| Exploiter | [2, 3, 4, 5] | Slow adaptation, high stability |
| Explorer | [5, 7, 9, 11] | Fast adaptation, responsive |

**Validation**:
- Exploiter beliefs should change slowly (< 0.5 levels per 10 ticks)
- Explorer beliefs should track ground truth within 20 ticks

### Phase 4: Learning Rate Calibration

**Goal**: Find learning rates that produce stable Q-value convergence.

**Method**: Sweep learning rates, measure Q-value stability.

| Parameter | Sweep Values |
|-----------|--------------|
| `learning_rate` | [0.05, 0.10, 0.15, 0.20, 0.25] |
| `epsilon` | [0.15, 0.25, 0.35, 0.45] |

**Interaction**: Test 2x2 grid of (learning_rate, epsilon) at extremes.

**Validation**:
- Q-values should stabilize (CV < 0.1) by tick 100
- No oscillation in final 50 ticks

### Phase 5: Trust Learning Rate Calibration

**Goal**: Find trust learning rates that differentiate agent types.

**Method**: Sweep trust learning rates, measure trust dynamics.

| Parameter | Sweep Values |
|-----------|--------------|
| `exploit_trust_lr` | [0.01, 0.02, 0.03, 0.05, 0.08] |
| `explor_trust_lr` | [0.03, 0.05, 0.08, 0.10, 0.12] |

**Constraint**: `explor_trust_lr` should be 1.5-3x `exploit_trust_lr`

**Validation**:
- Exploiter trust should be more stable than explorer trust
- Both should converge to reflect actual source quality

### Phase 6: Full Factorial Validation

After individual calibration, run full factorial on research variables:

```
ai_alignment: [0.1, 0.5, 0.9]
share_exploitative: [0.3, 0.5, 0.7]
= 9 conditions × 10 replications = 90 runs
```

**Success Criteria**:
1. Main effect of alignment on SECI (p < 0.05)
2. Main effect of alignment on MAE (p < 0.05)
3. Interaction: Exploiters show stronger alignment effect on SECI

---

## Recommended Calibration Experiment

See `calibration_experiments.py` for runnable code.

### Quick Calibration (30 minutes)

```bash
python calibration_experiments.py --quick
```
- 3 values per parameter
- 3 replications
- ~50 runs total

### Full Calibration (2-4 hours)

```bash
python calibration_experiments.py --full
```
- 5 values per parameter
- 5 replications
- ~200 runs total

### Output

Calibration produces:
1. `calibration_results.csv` - Raw data
2. `calibration_summary.md` - Recommended values
3. `calibration_plots/` - Visualization of sweeps

---

## Expected Calibrated Values

Based on theoretical analysis, expected final values:

| Parameter | Exploiter | Explorer | Rationale |
|-----------|-----------|----------|-----------|
| D | 1.5 ± 0.3 | 3.0 ± 0.5 | Paper baseline × 5 (scale 0-5) |
| δ | 18-22 | 6-10 | Sharp vs gradual |
| memory_size | 3-4 | 6-8 | Filter vs retain |
| trust_lr | 0.02-0.04 | 0.04-0.08 | 2x ratio |

---

## Troubleshooting

### Problem: No differentiation between agent types
- **Cause**: D values too similar or δ values too low
- **Fix**: Increase δ for exploiters, decrease for explorers

### Problem: Exploiters never accept anything
- **Cause**: D too low or δ too high
- **Fix**: Increase D or decrease δ for exploiters

### Problem: Explorers accept everything
- **Cause**: D too high or δ too low
- **Fix**: Decrease D or increase δ for explorers

### Problem: Q-values don't converge
- **Cause**: learning_rate too high or epsilon too high
- **Fix**: Reduce learning_rate, reduce epsilon after initial period

### Problem: Trust doesn't differentiate good/bad sources
- **Cause**: trust_lr too low or feedback too noisy
- **Fix**: Increase trust_lr, ensure feedback mechanism is working
