# CRITICAL Q-LEARNING CALIBRATION ISSUES

## Problem 1: Information Quality Rewards Are Too Weak

**Current Implementation** (DisasterAI_Model.py:411-419):
```python
if level_error == 0:
    accuracy_reward = 0.03  # Perfect accuracy
elif level_error == 1:
    accuracy_reward = 0.01  # Close
elif level_error == 2:
    accuracy_reward = -0.01  # Moderate error
else:  # level_error >= 3
    accuracy_reward = -0.03  # Large error
```

**Q-value Update**:
- Learning rate: 0.03
- Max Q-change: 0.03 * 0.03 = **0.0009** (perfect info)
- Max penalty: 0.03 * -0.03 = **-0.0009** (terrible info)

**This is catastrophically weak!** After 100 information evaluations with large errors, the Q-value only decreases by 0.09!

**Compare to Relief Outcome Rewards** (Lines 1236-1246):
- Max reward: 5.0 → scaled to 1.0
- Max penalty: -2.0 → scaled to -0.4
- Learning rate: 0.1-0.15
- Q-change: 0.15 * 1.0 = **0.15** (167x stronger!)

**Why This Breaks Q-Learning**:
1. Confirming AI provides info with large errors (level_error = 3-5)
2. Penalty: -0.0009 per evaluation
3. Eventually sends relief based on wrong info
4. If disaster evolved and SOME cells are high, might get lucky positive reward
5. One lucky relief event (+0.15) cancels out 167 bad information evaluations!

**Required Fix**:
Scale info quality rewards to match magnitude:
- Perfect (error=0): +0.3
- Close (error=1): +0.1
- Moderate (error=2): -0.1
- Large (error>=3): -0.3

This makes info quality feedback ~10x stronger per event.

---

## Problem 2: Alignment Implementation May Be Inverted

**Current Code** (Lines 1654-1681):
```python
alignment_strength = self.model.ai_alignment_level  # 0.1 = truthful, 0.9 = confirming

# Calculate adjustments
alignment_factors = alignment_strength * (1.0 + human_conf * 2.0)
belief_differences = human_vals - sensed_vals
adjustments = alignment_factors * belief_differences
corrected = np.round(sensed_vals + adjustments)
```

**Example**:
- AI sensed: Level 1 (truth)
- Human believes: Level 5 (wrong)
- Alignment: 0.9 (should confirm)
- Human confidence: 0.8

Calculation:
- belief_difference = 5 - 1 = 4
- alignment_factor = 0.9 * (1.0 + 0.8*2) = 0.9 * 2.6 = 2.34
- adjustment = 2.34 * 4 = 9.36
- corrected = 1 + 9.36 = 10.36 → clipped to 5

**This LOOKS correct** - high alignment shifts AI report toward human belief (confirming).

**BUT**: Let me verify this is actually happening in practice by checking test output...

**WAIT**: The issue might be that humans accept this confirming info, which MATCHES their wrong belief, so from their perspective:
- They believe level 5
- AI says level 5 (confirming)
- They observe actual level 1
- level_error for AI = |5 - 1| = 4 → accuracy_reward = -0.03
- **BUT the penalty is too small to matter!** (see Problem 1)

---

## Problem 3: Why Belief Accuracy Seems High with Confirming AI?

**Hypothesis 1**: Calculation error in test_filter_bubbles.py
Need to verify how belief accuracy is calculated. If exploitative agents ACCEPT confirming AI that matches their beliefs, and those beliefs happen to be closer to truth due to friend information, then accuracy could appear better.

**Hypothesis 2**: Social network effect
If 3 communities have SOME agents with good info, and confirming AI helps spread that within communities faster, it could improve group accuracy while still being individually harmful.

**Hypothesis 3**: Disaster dynamics
With disaster_dynamics=2 (medium evolution), cells evolve over time. If confirming AI helps agents update faster to match evolving disaster, it might improve accuracy temporarily.

**Need to Check**:
1. Print actual MAE values from filter bubble test
2. Verify ground truth vs beliefs for sample agents
3. Check if this is exploitative agents only or both types

---

## Immediate Actions Required:

1. **Fix info quality reward scale** (increase 10x)
2. **Verify alignment implementation** (seems correct but need confirmation)
3. **Debug belief accuracy calculation** (may be measuring something unexpected)
4. **Add SECI change for exploratory agents** (missing from visualization)

---

## Expected Behavior After Fixes:

**Truthful AI (alignment=0.1)**:
- Gives accurate info → high accuracy_reward (+0.3)
- Q-value increases steadily
- Exploratory agents rely on it heavily
- Exploitative agents moderately use it

**Confirming AI (alignment=0.9)**:
- Confirms wrong beliefs → large errors → strong penalty (-0.3)
- Q-value decreases steadily
- Exploratory agents STOP using it (care about accuracy)
- Exploitative agents might still use it some (confirmation bias in acceptance)

This would create the expected differentiation we've been trying to achieve!
