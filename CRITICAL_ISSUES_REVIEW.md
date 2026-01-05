# CRITICAL ISSUES - Model Logic Review

## Executive Summary

After comprehensive review of the codebase against requirements, **6 CRITICAL issues** were found that explain why results are bad:

1. **Explorers DON'T seek uncertain cells** - defeats their core purpose
2. **Missing latency/distance reward** - contradicts explicit requirements
3. **AI sensing noise logic backwards** - creates confusion in AI behavior
4. **Reward calculation doesn't incentivize exploration** - wrong weights
5. **Exploiters query mechanism needs refinement** - somewhat correct but could be better
6. **Q-learning mode updates potentially inconsistent for AI** - technical issue

---

## ISSUE #1: EXPLORERS DON'T SEEK UNCERTAIN CELLS ❌❌❌

### Requirements
> "Explorers seek information from UNCERTAIN cells and favour sources that provided accurate info."

### Current Implementation
```python
# Line 936-940 in seek_information()
else:  # Exploratory
    # CRITICAL FIX: Query about current vicinity so info can be evaluated when sensing
    interest_point = self.pos  # Query about where we are NOW
    query_radius = 2  # Match sensing radius for evaluation overlap
```

### Problem
- Explorers query about their **current position**, NOT uncertain cells
- The `find_exploration_targets()` function EXISTS (line 585-696) and correctly identifies uncertain cells
- BUT this function is NEVER USED for information seeking!
- Explorers are behaving like exploiters - just querying their immediate vicinity

### Impact
**CATASTROPHIC** - Explorers lose their defining characteristic:
- Can't learn which sources provide good info about uncertain areas
- Can't discover new disaster zones
- Defeats the entire purpose of having two agent types
- Makes it impossible to test filter bubble dynamics properly

### Fix Required
```python
else:  # Exploratory
    # Find uncertain cells to explore
    self.find_exploration_targets(num_targets=3)

    if self.exploration_targets:
        # Select most uncertain cell to query about
        interest_point = self.exploration_targets[0]
    else:
        interest_point = self.pos  # Fallback

    query_radius = 3  # Wider radius for exploration
```

---

## ISSUE #2: MISSING LATENCY/DISTANCE REWARD COMPONENT ❌❌

### Requirements
> "q-learning algorithm based on relief sent (higher latency, greater impact) and information received"

### Current Implementation
```python
# Line 1276-1287 in process_reward()
if actual_level == 5:
    cell_reward = 5.0  # Perfect targeting
elif actual_level == 4:
    cell_reward = 3.0  # Very good targeting
# ... etc - purely based on disaster level
```

### Problem
- Reward is ONLY based on actual disaster level
- NO component for distance/latency
- Requirements explicitly state "higher latency, greater impact"
- Agents should get MORE reward for correctly targeting distant cells (harder/riskier)

### Impact
**HIGH** - Incorrect learning incentives:
- Agents don't learn value of sources that help find distant disasters
- No incentive to explore far from current position
- Reduces diversity of relief distribution
- Makes all agents cluster around easily-found disasters

### Fix Required
```python
# Calculate distance from agent to cell (proxy for latency)
distance = math.sqrt((cell[0] - self.pos[0])**2 + (cell[1] - self.pos[1])**2)
max_distance = math.sqrt(self.model.width**2 + self.model.height**2)
normalized_distance = distance / max_distance  # [0, 1]

# Base reward from disaster level
if actual_level == 5:
    base_reward = 5.0
elif actual_level == 4:
    base_reward = 3.0
elif actual_level == 3:
    base_reward = 1.5
elif actual_level == 2:
    base_reward = 0.0
elif actual_level == 1:
    base_reward = -1.0
else:
    base_reward = -2.0

# Add latency bonus (higher for distant cells when correct)
if actual_level >= 3:  # Only for successful relief
    latency_bonus = normalized_distance * 2.0  # Up to +2.0 for max distance
    cell_reward = base_reward + latency_bonus
else:
    # Penalty is higher for distant wrong cells (wasted resources)
    latency_penalty = normalized_distance * 1.0
    cell_reward = base_reward - latency_penalty
```

---

## ISSUE #3: AI SENSING NOISE LOGIC IS BACKWARDS ❌

### Current Implementation
```python
# Line 1524-1531 in AIAgent.sense_environment()
# When alignment is low, AI should be more truthful/accurate
noise_prob = 0.1  # Default 10% chance of noise
if hasattr(self.model, 'ai_alignment_level'):
    # Reduce noise when alignment is low (more truthful)
    noise_prob = 0.1 * self.model.ai_alignment_level

if random.random() < noise_prob:
    value = max(0, min(5, value + random.choice([-1, 1])))
```

### Problem
- AI adds noise DURING SENSING based on alignment
- Comment says "reduce noise when alignment is low" but formula does opposite
- `noise_prob = 0.1 * alignment_level` means:
  - alignment=0 (truthful): noise_prob=0 ✓ (correct)
  - alignment=1 (confirming): noise_prob=0.1 ✓ (correct)
- Actually the formula IS correct! But then...
- AI also adjusts reports based on alignment (line 1695-1722)
- This creates DOUBLE adjustment - sensing AND reporting

### Impact
**MEDIUM** - Confusing behavior:
- Truthful AI (low alignment) senses accurately AND reports accurately ✓
- Confirming AI (high alignment) senses with noise AND reports with bias
- The double adjustment might be too strong
- Makes it harder for agents to distinguish AI quality

### Fix Required
Remove noise from sensing, keep only in reporting:
```python
# AI should ALWAYS sense truth accurately
# Bias is applied only in report_beliefs() based on alignment
# This makes AI behavior clearer and more testable

# Remove lines 1524-1531
# Keep only the report adjustment logic (lines 1695-1722)
```

---

## ISSUE #4: REWARD CALCULATION WEIGHTS ARE COUNTERPRODUCTIVE ❌

### Current Implementation
```python
# Line 1328-1338 in process_reward()
if self.agent_type == "exploratory":
    # Explorers care almost exclusively about actual accuracy
    avg_actual_reward = sum(cell_rewards) / len(cell_rewards)
    correct_ratio = correct_in_batch / len(cell_rewards) if cell_rewards else 0
    batch_reward = 0.8 * avg_actual_reward + 0.2 * (correct_ratio * 5.0)  # 80% actual, 20% confirm
else:
    # Exploiters care much more about validation of beliefs than actual accuracy
    correct_ratio = correct_in_batch / len(cell_rewards) if cell_rewards else 0
    avg_actual_reward = sum(cell_rewards) / len(cell_rewards)
    batch_reward = 0.2 * avg_actual_reward + 0.8 * (correct_ratio * 5.0)  # 20% actual, 80% confirm
```

### Problem
- "Confirmation" is defined as `actual_level >= 3` (line 1268)
- This is NOT confirmation of BELIEFS, it's confirmation of SUCCESS
- Both metrics are essentially measuring the same thing: "did we help?"
- The weights don't actually create different incentives
- Explorers should be rewarded for REDUCING UNCERTAINTY, not just accuracy
- Exploiters should be rewarded for CONFIRMING THEIR BELIEFS (belief matches reality)

### Impact
**MEDIUM** - Agents don't develop distinct strategies:
- Both agent types learn similar patterns
- No differentiation in learning objectives
- Filter bubble effects won't emerge properly

### Fix Required
```python
# For exploratory agents: reward reducing uncertainty
if self.agent_type == "exploratory":
    # Reward based on: accuracy + uncertainty reduction
    # Track cells that had low confidence but turned out to need relief
    uncertainty_reward = 0
    for cell, belief_level in cells_and_beliefs:
        prior_confidence = self.beliefs[cell].get('confidence', 0.5)
        uncertainty = 1.0 - prior_confidence
        actual_level = self.model.disaster_grid[cell[0], cell[1]]
        if actual_level >= 3:  # Correctly found high-need cell
            uncertainty_reward += uncertainty * 2.0  # Bonus for uncertain->correct

    batch_reward = 0.6 * avg_actual_reward + 0.4 * uncertainty_reward

else:  # exploitative
    # Reward based on: belief confirmation + friend validation
    confirmation_reward = 0
    for cell, belief_level in cells_and_beliefs:
        actual_level = self.model.disaster_grid[cell[0], cell[1]]
        belief_accuracy = 1.0 - abs(belief_level - actual_level) / 5.0
        confidence = self.beliefs[cell].get('confidence', 0.5)
        # High reward for: high confidence + accurate belief
        confirmation_reward += belief_accuracy * confidence

    batch_reward = 0.3 * avg_actual_reward + 0.7 * confirmation_reward
```

---

## ISSUE #5: EXPLOITERS' QUERY MECHANISM PARTIALLY CORRECT ⚠️

### Current Implementation
```python
# Line 886-934 in seek_information()
if self.agent_type == "exploitative":
    self.find_believed_epicenter()
    interest_point = self.believed_epicenter
    query_radius = 2
```

### Analysis
- Exploiters query about their believed epicenter (highest believed level)
- This is PARTIALLY correct for "confirm existing beliefs"
- Requirements say: "seek to CONFIRM their existing beliefs and favour their friends"

### Issue
- Querying about the highest-level cell is good
- BUT exploiters should also query about cells where they have HIGH CONFIDENCE
- Current logic focuses on highest LEVEL, not highest CONFIDENCE
- For true confirmation bias, they should prefer querying about strongly-held beliefs

### Impact
**LOW-MEDIUM** - Behavior is close to requirements but not optimal:
- Exploiters do seek confirmation
- But they're not specifically targeting their most confident beliefs
- Could reduce filter bubble effects

### Fix Suggested
```python
if self.agent_type == "exploitative":
    # Find cells with BOTH high belief level AND high confidence
    # This represents strong existing beliefs that need "confirmation"
    high_confidence_beliefs = []
    for cell, belief_info in self.beliefs.items():
        if isinstance(belief_info, dict):
            level = belief_info.get('level', 0)
            confidence = belief_info.get('confidence', 0)
            # Score = level * confidence (want both high)
            if level >= 2 and confidence >= 0.5:
                score = level * confidence
                high_confidence_beliefs.append((cell, score))

    if high_confidence_beliefs:
        # Sort by score and pick highest
        high_confidence_beliefs.sort(key=lambda x: x[1], reverse=True)
        interest_point = high_confidence_beliefs[0][0]
    else:
        # Fallback to believed epicenter
        self.find_believed_epicenter()
        interest_point = self.believed_epicenter

    query_radius = 2
```

---

## ISSUE #6: Q-LEARNING MODE UPDATE LOGIC POTENTIALLY INCONSISTENT ⚠️

### Current Implementation
```python
# Line 1071: Set mode in token tracker
self.tokens_this_tick = {chosen_mode: 1}  # chosen_mode = "ai"

# Line 1108: Select specific AI
source_id = self.choose_best_ai_for_query(interest_point, query_radius)  # returns "A_0"

# Line 1211: Get mode from tokens
responsible_mode = list(self.tokens_this_tick.keys())[0]  # Should be "ai"

# Line 1365-1372: Update mode Q-value
if mode in self.q_table:  # Check if "ai" is in Q-table
    old_mode_q = self.q_table[mode]
    # ... update
```

### Analysis
Looking at initialization (line 110-120):
```python
self.q_table["self_action"] = 0.0
self.q_table["human"] = 0.0
self.q_table["ai"] = 0.0  # ✓ "ai" mode exists
```

### Status
Actually this looks CORRECT:
- Mode "ai" is initialized in Q-table (line 115)
- `chosen_mode = "ai"` is stored (line 1071)
- Update checks `if mode in self.q_table` (line 1365)
- Should work correctly

### Potential Issue
The comment on line 1364 says "fixes mode vs source ID mismatch" but for AI:
- Mode is "ai" (generic)
- But when agents choose which AI to query, it uses `choose_best_ai_for_query()`
- This is actually GOOD - agents learn value of "AI as a category"
- Then specific AI is chosen based on knowledge coverage

### Conclusion
**NO FIX NEEDED** - Logic is correct as-is

---

## SUMMARY OF REQUIRED FIXES

| Issue | Priority | Impact | Fix Complexity |
|-------|----------|--------|----------------|
| #1: Explorers don't seek uncertain cells | **CRITICAL** | Breaks core model | Medium |
| #2: Missing latency reward | **CRITICAL** | Wrong incentives | Medium |
| #3: AI sensing noise | **MEDIUM** | Confusing behavior | Easy |
| #4: Reward weights | **MEDIUM** | Weak differentiation | Medium |
| #5: Exploiter query logic | **LOW** | Minor optimization | Easy |
| #6: Q-learning mode update | **NONE** | Working correctly | N/A |

## RECOMMENDED FIX ORDER

1. **Issue #1** - Most critical, breaks explorer behavior completely
2. **Issue #2** - Critical for correct Q-learning
3. **Issue #3** - Simple fix, clarifies AI behavior
4. **Issue #4** - Improves agent differentiation
5. **Issue #5** - Optional refinement

---

## ADDITIONAL OBSERVATIONS

### What's Working Well ✓
- Q-learning infrastructure is solid (fixes from previous PRs working)
- Trust update mechanisms are correct
- AI alignment implementation in reporting is good (despite sensing noise issue)
- Metrics calculation (SECI/AECI) is comprehensive
- Social network structure is correct

### What Needs Attention
- Explorer behavior completely broken (Issue #1)
- Reward structure missing key component (Issue #2)
- Agent differentiation could be stronger (Issue #4)

### Testing Recommendations
After fixes, verify:
1. Explorers query about uncertain cells (not current position)
2. Distant correct relief gets higher rewards than nearby
3. AI Q-values update properly for both agent types
4. Explorers develop preference for accurate sources
5. Exploiters develop preference for confirming sources
6. Filter bubble metrics show meaningful differences between agent types
