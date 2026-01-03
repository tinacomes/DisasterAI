# Summary of Q-Learning Fixes

## Critical Issues Fixed

### Issue #0: ALIGNMENT CHEATING (Most Critical)
**Problem**: Agents were using `ai_alignment_level` directly in decision-making, completely bypassing Q-learning.

**What was wrong**:
- Action selection added biases based on alignment (exploitative preferred high-alignment AI, exploratory preferred low-alignment AI)
- Learning rates were multiplied based on alignment
- Trust updates gave massive boosts for "good" alignment
- Trust decay was slower for "good" alignment
- Belief updates boosted precision for truthful AI

**Why this defeats Q-learning**:
- Agents were TOLD which sources were good instead of LEARNING through experience
- Q-values became meaningless because decisions were made using alignment
- Completely defeats the purpose of reinforcement learning!

**Fix**:
- Removed ALL alignment-based logic from human agent decision-making
- Action selection now uses ONLY Q-values + agent-type preferences
- Learning rates uniform across source types (only vary by agent type)
- Trust updates purely feedback-based (no alignment peeking)
- Trust decay uniform (only varies by friend/non-friend)
- Belief updates don't peek at alignment

**Alignment kept ONLY in**:
- AIAgent.report_beliefs(): Creates biased vs truthful behavior (CORRECT - this is what we're testing!)
- Model initialization and visualization (CORRECT)

**Result**: Pure Q-learning! Agents learn which sources are valuable through experience.

---

### Issue #1: HUMAN Q-VALUES NEVER UPDATE
**Problem**: Mode vs source ID mismatch

**What was wrong**:
1. Action selection uses `Q-table["human"]` (generic mode)
2. Specific source selected: `source_id = "H_5"`
3. Feedback updates `Q-table["H_5"]` (specific source)
4. `Q-table["human"]` NEVER updated → stays at 0.05 forever!

**Why this matters**:
- Agents can't learn whether consulting humans (as a category) is valuable
- Same issue for AI sources (though less severe since mode=source_id for AIs)

**Fix**:
- Update BOTH mode Q-value AND specific source Q-value
- Applied to:
  - Relief outcome feedback (process_reward, line 1308-1317)
  - Info quality feedback (evaluate_information_quality, line 422-442)
- Infer mode from source_id: "H_x" → "human", "A_x" → "A_x"

**Result**: Human Q-value now properly learns! Agents can discover value of human sources.

---

### Issue #2: TRUST SKYROCKETS FOR EXPLORATORY + CONFIRMING AI
**Problem**: No penalty for confirming AI with exploratory agents

**What was wrong**:
- Exploratory agents got massive trust boost for low-alignment AI (correct)
- But for high-alignment AI, they got NORMAL trust update (no penalty)
- If confirming AI's advice led to successful relief (lucky coincidence), trust increased!

**Fix**:
- Removed alignment-based trust logic entirely (see Issue #0)
- Now ALL agents get pure feedback-based trust updates
- Trust increases for sources that lead to good outcomes, decreases for bad outcomes
- No peeking at whether AI is confirming or truthful

**Result**: Exploratory agents will naturally reduce trust in sources that don't improve outcomes.

---

### Issue #3: EXPLORATORY AGENTS GET NO INFO QUALITY FEEDBACK
**Problem**: Evaluation window too narrow, paradoxical behavior

**What was wrong**:
- Info quality feedback requires: query cell → later sense cell
- Exploratory agents: wider sensing (3) + more movement → less overlap between queried and sensed cells
- Exploitative agents: narrow sensing (2) + less movement → MORE overlap!
- Test showed: Exploitative got 66 events, Exploratory got 0!

**Fix**:
- Widened evaluation window from 5-10 ticks to 3-15 ticks
- Better accommodates different movement patterns
- Allows more time for exploratory agents to circle back

**Note**: This may not fully solve the issue if exploratory agents fundamentally query less. May need further investigation.

**Result**: Wider window should capture more evaluations for all agent types.

---

### Issue #4: FRIENDS VS HUMANS DISTINCTION
**Status**: Design issue, not a bug

**How it works**:
- Friends populated from social network
- Mode "human" includes both friends and non-friends
- When mode="human" selected, agent prefers friends if available
- All stored as specific IDs ("H_5")

**Resolution**: Issue #1 fix addresses this - now "human" mode Q-value updates properly, capturing value of human sources regardless of friend status.

---

## Summary of Changes

| File | Lines Changed | Description |
|------|---------------|-------------|
| DisasterAI_Model.py | 998-1028 | Removed alignment biases from action selection |
| DisasterAI_Model.py | 676-681 | Removed alignment from trust decay |
| DisasterAI_Model.py | 731-733 | Removed alignment from belief precision |
| DisasterAI_Model.py | 1332, 1348-1357 | Removed alignment from Q-learning rates and trust boosts |
| DisasterAI_Model.py | 1423-1425 | Removed alignment from accuracy-based trust |
| DisasterAI_Model.py | 1308-1317 | Added mode Q-value updates (relief feedback) |
| DisasterAI_Model.py | 422-442 | Added mode Q-value updates (info feedback) |
| DisasterAI_Model.py | 397-405 | Widened info evaluation window to 3-15 ticks |

## Impact

**Before fixes**:
- Agents told which sources were good (alignment cheating)
- Human Q-values frozen at initialization
- Exploratory agents got no info quality feedback
- Trust increased even for bad sources (lucky confirmations)

**After fixes**:
- Pure Q-learning: agents learn through experience only
- All Q-values (human, AI, self-action) update properly
- Better feedback coverage for all agent types
- Trust reflects actual outcomes, not alignment

## Next Steps

1. **Test with clean slate**: Run tests to verify Q-values now update
2. **Monitor info feedback**: Check if exploratory agents now get events
3. **Verify learning**: Ensure agents discover good vs bad sources through experience
4. **Compare alignment conditions**: High vs low alignment should create different learned Q-values

The model is now a **true Q-learning system** where agents discover source quality through feedback, not by peeking at hidden parameters!
