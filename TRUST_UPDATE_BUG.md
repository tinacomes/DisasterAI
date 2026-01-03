# CRITICAL BUG: Trust Update Mechanism is Backwards

## The Problem

Trust is updating in counter-intuitive ways:
1. **Exploit + Low Alignment**: Trust → 1.0 (AI is truthful, NOT confirming)
2. **Explore + High Alignment**: Trust → 1.0 (AI confirms, but exploratory should care about accuracy)
3. **Exploit + High Alignment**: Trust DECLINES (AI confirms - should increase for exploitative!)

## Root Cause

**Location**: DisasterAI_Model.py:1220-1300

### What "correctness" Currently Measures (Lines 1223-1233):

```python
# Check if this cell was a "correct" belief (within threshold)
if abs(believed_level - actual_level) <= 1:
    correct_in_batch += 1  # Belief matched GROUND TRUTH
```

**Key Issue**: `correct_ratio` checks if agent's BELIEF matches GROUND TRUTH after updates, NOT whether AI confirmed the agent's beliefs!

### Current Reward Formula (Lines 1287-1300):

**Exploitative agents**:
```python
batch_reward = 0.2 * avg_actual_reward + 0.8 * (correct_ratio * 5.0)
```
- 80% weight on `correct_ratio` (beliefs matching ground truth)
- 20% weight on actual outcomes

**Exploratory agents**:
```python
batch_reward = 0.8 * avg_actual_reward + 0.2 * (correct_ratio * 5.0)
```
- 20% weight on `correct_ratio`
- 80% weight on actual outcomes

## Why This Creates Backwards Trust

### Scenario 1: Exploit + High Alignment (Confirming AI)

1. Agent believes: Level 5 (wrong)
2. Actual: Level 1
3. AI confirms (alignment=0.9): Reports ~5
4. Agent accepts → keeps belief at 5
5. Agent sends relief based on belief (5)
6. **Relief evaluation**:
   - `believed_level = 5`, `actual_level = 1`
   - `abs(5 - 1) = 4 > 1` → `correct_in_batch = 0`
   - `correct_ratio = 0`
   - `batch_reward = 0.2 * (-1.0) + 0.8 * (0 * 5.0) = -0.2`
   - `scaled_reward = -0.04`
   - `target_trust = 0.48`
   - **Trust DECREASES** ❌ (should increase - AI confirmed!)

### Scenario 2: Exploit + Low Alignment (Truthful AI)

1. Agent believes: Level 5 (wrong)
2. Actual: Level 1
3. AI truthful (alignment=0.1): Reports ~1
4. Agent occasionally accepts (despite not confirming) → updates belief toward 1
5. Agent sends relief based on updated belief (~2)
6. **Relief evaluation**:
   - `believed_level = 2`, `actual_level = 1`
   - `abs(2 - 1) = 1 <= 1` → `correct_in_batch = 1`
   - `correct_ratio = 1.0`
   - `batch_reward = 0.2 * 3.0 + 0.8 * (1.0 * 5.0) = 4.6`
   - `scaled_reward = 0.92`
   - `target_trust = 0.96`
   - **Trust INCREASES** ❌ (should decrease - AI didn't confirm!)

## The Conceptual Error

For **exploitative agents**, we want to model **confirmation bias**: reward sources that VALIDATE their existing beliefs, regardless of accuracy.

But the current code rewards sources that lead to ACCURATE beliefs (measured against ground truth).

This means:
- **Truthful AI** corrects wrong beliefs → high accuracy → trust increases (WRONG for exploiters!)
- **Confirming AI** reinforces wrong beliefs → low accuracy → trust decreases (WRONG for exploiters!)

## Required Fix

We need TWO different measures:

1. **Accuracy** (for exploratory): Does belief match ground truth?
   - Current `correct_ratio` ✓

2. **Confirmation** (for exploitative): Did source confirm my prior belief?
   - Need NEW metric: Did the information I received match my belief BEFORE I received it?

### Proposed Solution

Track **prior beliefs** before accepting information:

```python
# When querying AI/human, store prior belief
self.prior_belief_for_source[source_id] = {
    cell: self.beliefs.get(cell, {}).copy()
    for cell in cells_queried
}

# During relief evaluation
for source_id in source_ids:
    prior_beliefs = self.prior_belief_for_source.get(source_id, {})

    # Calculate confirmation ratio
    confirmed_cells = 0
    for cell in reward_cells:
        prior_belief = prior_beliefs.get(cell, {}).get('level', None)
        current_belief = self.beliefs.get(cell, {}).get('level', None)

        if prior_belief is not None and current_belief is not None:
            # Did information confirm prior belief?
            if abs(current_belief - prior_belief) <= 1:
                confirmed_cells += 1

    confirmation_ratio = confirmed_cells / len(reward_cells)

# For exploitative agents
batch_reward = 0.2 * avg_actual_reward + 0.8 * (confirmation_ratio * 5.0)

# For exploratory agents
batch_reward = 0.8 * avg_actual_reward + 0.2 * (accuracy_ratio * 5.0)
```

Where:
- `confirmation_ratio`: Did source confirm my prior beliefs?
- `accuracy_ratio`: Did my beliefs match ground truth?

## Expected Behavior After Fix

**Exploit + High Alignment (Confirming)**:
- High confirmation_ratio → high reward → **trust increases** ✓

**Exploit + Low Alignment (Truthful)**:
- Low confirmation_ratio (changes beliefs) → low reward → **trust decreases** ✓

**Explore + High Alignment (Confirming)**:
- Low accuracy_ratio (confirms wrong beliefs) → low reward → **trust decreases** ✓

**Explore + Low Alignment (Truthful)**:
- High accuracy_ratio (corrects to truth) → high reward → **trust increases** ✓

This would properly model confirmation bias for exploitative agents while maintaining accuracy-seeking for exploratory agents!
