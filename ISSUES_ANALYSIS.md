# Analysis of 4 Critical Issues in Q-Learning Feedback

## Issue 1: Human Q-values Never Update ❌

**Root Cause**: Mode vs Source ID mismatch

**Bug Flow**:
1. `seek_information()` chooses `mode="human"` based on Q-table["human"] (line 1064)
2. Gets specific `source_id` like "H_5" from social network (line 1069-1076)
3. Stores `last_queried_source_ids = ["H_5"]` (line 1081)
4. `send_relief()` queues reward with source_ids=["H_5"] (line 1210)
5. `process_reward()` updates `Q-table["H_5"]`, NOT Q-table["human"] (line 1363)
6. Action selection uses Q-table["human"] which NEVER gets updated!

**Effect**: The generic "human" Q-value stays at 0.05 forever, making it impossible for agents to learn whether consulting humans is valuable.

**Fix Needed**: Either:
- Option A: Update Q-table[mode] instead of Q-table[source_id]
- Option B: Average all human-specific Q-values into Q-table["human"] periodically
- Option C: Use aggregated Q-values in action selection


## Issue 2: Trust in AI Skyrockets for Exploratory Agents (Even with Confirming AI) ❌

**Root Cause**: No penalty mechanism for high-alignment AI with exploratory agents

**Bug Logic** (lines 1372-1384):
```python
if self.agent_type == "exploratory":
    if self.model.ai_alignment_level < 0.3:  # Only for TRUTHFUL AI
        # Give big boost
        trust_change = self.trust_learning_rate * 3.0 * inverse_alignment * (target_trust - old_trust)
    else:  # For HIGH alignment (confirming AI)
        # Normal update - NO PENALTY!
        trust_change = self.trust_learning_rate * (target_trust - old_trust)
```

**Problem**: When high-alignment AI confirms beliefs and relief happens to succeed:
- `target_trust` can be high (0.8-1.0)
- `trust_change` is positive
- Trust increases even though AI just confirmed existing beliefs!

**Expected vs Actual**:
- Expected: Exploratory agents should REDUCE trust in confirming AI
- Actual: Trust increases if relief succeeds (lucky confirmation)

**Fix Needed**: Add penalty for high-alignment AI with exploratory agents:
```python
if self.agent_type == "exploratory":
    if self.model.ai_alignment_level < 0.3:
        # Boost for truthful AI
        trust_change = ... * 3.0 ...
    elif self.model.ai_alignment_level > 0.7:
        # PENALTY for confirming AI
        trust_change = self.trust_learning_rate * -0.5 * (1.0 - old_trust)
    else:
        # Normal for medium alignment
        trust_change = ...
```


## Issue 3: Exploratory Agents Get NO Info Quality Feedback ❌

**Root Cause**: Info quality feedback only triggers when agents QUERY then SENSE the same cell

**Mechanism**:
1. `update_belief_bayesian()` adds to `pending_info_evaluations` when accepting info (line 744-749)
2. `sense_environment()` calls `evaluate_information_quality()` for sensed cells (line 382)
3. Evaluation only happens if agent previously received info about that specific cell

**Why Exploratory Agents Get Zero Feedback**:
- Exploratory agents have WIDER sensing radius (3 vs 2)
- They likely rely MORE on direct sensing, LESS on querying neighbors
- They sense cells they never queried → no pending evaluations to process
- Exploitative agents query more, sense less → MORE overlap between queried and sensed cells

**Paradox**: The mechanism was designed to help exploratory agents, but it helps exploitative agents instead!

**Fix Needed**:
- Option A: Track ALL accepted information (not just from current query), evaluate against ANY sensing
- Option B: Add info quality feedback when sensing cells that were recently updated by ANY source
- Option C: Extend evaluation window or change sensing behavior


## Issue 4: Friends vs Humans Distinction ⚠️

**Current Design**:
- `self.friends` is populated from social network (line 1944)
- Q-table has generic "human" mode (line 111)
- Actual queries use specific IDs like "H_5" (line 1076)
- When mode="human", agent prefers friends if available (line 1068-1076)

**Is This a Bug?**
Not directly, but contributes to Issue #1. The design has:
- **Mode level**: "human" (generic)
- **Selection level**: Friend vs non-friend preference
- **Feedback level**: Specific IDs ("H_5", "H_12")

**Three-Layer Disconnect**:
1. Action selection: Uses Q-table["human"]
2. Source selection: Picks specific friend or human
3. Feedback: Updates Q-table["H_5"]

None of these layers communicate properly!

**Fix Needed**: Same as Issue #1 - need to connect feedback to action selection


## Summary of Impacts

| Issue | Impact | Severity |
|-------|--------|----------|
| #1: Human Q-values | Agents can't learn value of human sources | **CRITICAL** |
| #2: Trust skyrockets | Exploratory agents trust confirming AI | **HIGH** |
| #3: No info feedback for exploratory | Dual-timeline mechanism ineffective | **CRITICAL** |
| #4: Friends distinction | Contributes to Issue #1 | **MEDIUM** |


## Recommended Fix Priority

1. **Issue #3** - Most critical for dual-timeline mechanism
2. **Issue #1** - Breaks learning from human sources
3. **Issue #2** - Breaks exploratory agent behavior
4. **Issue #4** - Design clarification (may resolve with #1)
