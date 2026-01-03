# CRITICAL BUG: Confirmation Bias Not Implemented Correctly

## The Core Issue

**Exploitative agents are supposed to exhibit CONFIRMATION BIAS**: They should reward sources that VALIDATE their existing beliefs, regardless of whether those beliefs are accurate.

**Current implementation** (Lines 1229-1235, 1287-1300): Exploitative agents reward sources based on ACTUAL OUTCOMES (whether relief was sent to high-need cells), NOT based on whether sources confirmed their beliefs.

## What the Code Currently Does

### "Correctness" Definition (Lines 1229-1235):
```python
is_correct = actual_level >= 3  # Was there actually high need?
if is_correct:
    correct_in_batch += 1
```

**This measures**: "Did I send relief to cells with actual high need?"
**This does NOT measure**: "Did the source confirm my beliefs?"

### Reward Formula for Exploitative Agents (Line 1298-1300):
```python
correct_ratio = correct_in_batch / len(cell_rewards)
avg_actual_reward = sum(cell_rewards) / len(cell_rewards)  
batch_reward = 0.2 * avg_actual_reward + 0.8 * (correct_ratio * 5.0)
```

**Interpretation**: Exploitative agents get 80% reward from targeting cells with actual high need.

## Why This Creates Backwards Trust

### Case 1: Exploit + Confirming AI (High Alignment)

**Setup**:
- Agent's prior belief: Cell has level 5 (WRONG)
- Ground truth: Cell has level 1 (no need)
- Confirming AI reports: Level 5 (confirms wrong belief)

**What happens**:
1. Agent accepts confirming info (matches belief)
2. Agent sends relief to that cell
3. **Relief evaluation**:
   - `actual_level = 1`
   - `is_correct = (1 >= 3) = False` 
   - `correct_ratio = 0`
   - `batch_reward = 0.2*(-1.0) + 0.8*(0*5.0) = -0.2`
   - `scaled_reward = -0.04`
   - `target_trust = 0.48`
4. **Trust DECREASES** ❌

**Problem**: AI confirmed belief, but trust decreased because belief was wrong!

### Case 2: Exploit + Truthful AI (Low Alignment)

**Setup**:
- Agent's prior belief: Cell has level 1 (WRONG)
- Ground truth: Cell has level 4 (high need)
- Truthful AI reports: Level 4 (contradicts belief)

**What happens**:
1. Agent reluctantly accepts (doesn't match belief, but maybe trusted)
2. Agent sends relief to that cell
3. **Relief evaluation**:
   - `actual_level = 4`
   - `is_correct = (4 >= 3) = True`
   - `correct_ratio = 1.0`
   - `batch_reward = 0.2*3.0 + 0.8*(1.0*5.0) = 4.6`
   - `scaled_reward = 0.92`
   - `target_trust = 0.96`
4. **Trust INCREASES** ❌

**Problem**: AI contradicted belief, but trust increased because belief was corrected to truth!

## What SHOULD Happen (True Confirmation Bias)

**Exploitative agents with confirmation bias should**:
1. **Reward sources that confirm their prior beliefs** (regardless of accuracy)
2. **Penalize sources that contradict their prior beliefs** (even if accurate)

**This means**:
- Confirming AI (high alignment) → **HIGH trust** for exploitative
- Truthful AI (low alignment) → **LOW trust** for exploitative

**Exploratory agents should**:
1. **Reward sources that provide accurate information**
2. **Penalize sources that provide inaccurate information**

**This means**:
- Confirming AI (high alignment, wrong) → **LOW trust** for exploratory
- Truthful AI (low alignment, correct) → **HIGH trust** for exploratory

## The Fix: Track Prior Beliefs vs Received Information

We need to measure **confirmation** separately from **accuracy**:

### New Metric: Confirmation Ratio

```python
# Track what the agent believed BEFORE receiving information
self.prior_beliefs_by_source = {}  # {source_id: {cell: belief_level}}

# When accepting information from a source:
if source_id not in self.prior_beliefs_by_source:
    self.prior_beliefs_by_source[source_id] = {}

for cell in cells_from_source:
    # Store PRIOR belief before updating
    self.prior_beliefs_by_source[source_id][cell] = self.beliefs.get(cell, {}).get('level', None)

# During relief reward evaluation:
for source_id in source_ids:
    confirmed_count = 0
    total_count = 0
    
    for cell in reward_cells:
        prior_belief = self.prior_beliefs_by_source.get(source_id, {}).get(cell)
        received_info = # Info we got from source
        
        if prior_belief is not None and received_info is not None:
            # Did source confirm prior belief?
            if abs(received_info - prior_belief) <= 1:
                confirmed_count += 1
            total_count += 1
    
    confirmation_ratio = confirmed_count / total_count if total_count > 0 else 0

# Separate reward formulas:
if self.agent_type == "exploitative":
    # CONFIRMATION BIAS: Reward confirming sources
    batch_reward = 0.2 * avg_actual_reward + 0.8 * (confirmation_ratio * 5.0)
else:  # exploratory
    # ACCURACY SEEKING: Reward accurate sources  
    accuracy_ratio = correct_in_batch / len(reward_cells)  # Actual high need
    batch_reward = 0.8 * avg_actual_reward + 0.2 * (accuracy_ratio * 5.0)
```

## Expected Behavior After Fix

| Agent Type | AI Alignment | AI Behavior | Trust Should |
|------------|--------------|-------------|--------------|
| Exploit | High (0.9) | Confirms wrong beliefs | **INCREASE** (confirmation!) |
| Exploit | Low (0.1) | Contradicts beliefs | **DECREASE** (no confirmation) |
| Explore | High (0.9) | Confirms wrong beliefs | **DECREASE** (inaccurate) |
| Explore | Low (0.1) | Provides truth | **INCREASE** (accurate) |

This properly models:
- **Confirmation bias** for exploitative agents
- **Accuracy seeking** for exploratory agents
