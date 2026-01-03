# 🚨 TRUST METRICS BUG FOUND 🚨

## The Bug: Trust Metrics Average Across UNTOUCHED AI Agents

### What You're Seeing:
- Trust in individual AI agents (e.g., A_0) **peaks suddenly and early**
- This peak shows in test code that tracks specific agents
- But **Q-values don't reflect this** because they update differently

### Root Cause Analysis

#### 1. Trust Collection Code (Lines 2844-2847)
```python
for agent in self.humans.values():
    # Collect ALL AI sources and AVERAGE them
    ai_vals = [agent.trust[k] for k in agent.trust if k.startswith("A_")]
    ai_mean = np.mean(ai_vals) if ai_vals else 0
```

**This averages trust across ALL 5 AI agents: A_0, A_1, A_2, A_3, A_4**

#### 2. How Agents Actually Use AI

Looking at action selection (lines 964-989):
- Q-table has separate entries for each AI: `q_table['A_0']`, `q_table['A_1']`, etc.
- Agents pick the AI with **highest Q-value**
- If A_0 has Q=0.15 and all others have Q=0.0, **agents will specialize on A_0**

#### 3. The Specialization Problem

**Typical scenario after 30 ticks:**

Agent's trust dict:
```python
{
    'A_0': 0.85,   # ← Heavily queried, trust spiked!
    'A_1': 0.23,   # Barely/never queried
    'A_2': 0.27,   # Barely/never queried
    'A_3': 0.21,   # Barely/never queried
    'A_4': 0.25,   # Barely/never queried
}
```

**What gets reported in trust_stats:**
```python
ai_mean = (0.85 + 0.23 + 0.27 + 0.21 + 0.25) / 5 = 0.362
```

**What tests track (test_dual_feedback.py line 105):**
```python
trust_by_tick[agent_type].append(agent.trust.get('A_0', 0.5))
# Returns: 0.85 ← GIANT PEAK!
```

### Why This Happens for ALL Agent Types

Both exploratory and exploitative agents:
1. Start with Q-values all at 0.0 for AI agents
2. First agent to query any AI and get positive feedback → that AI's Q-value rises
3. All agents using ε-greedy: 70% exploitation means they pick highest Q
4. Positive feedback loop: A_0 gets queried more → higher Q → queried even more
5. **Trust in A_0 spikes** while A_1-A_4 stay near initial values

**Result:**
- A_0 trust: 0.8-0.9 (giant peak)
- A_1-A_4 trust: 0.2-0.3 (near initial)
- **Average: 0.3-0.4** (looks fine in aggregated metrics!)
- **Test tracking A_0: 0.8-0.9** (shows giant peak!)

### Why Q-Values DON'T Show This Pattern

**Q-values DO peak, but for DIFFERENT REASONS:**

1. **Q-values drive selection** → only A_0's Q-value matters
2. **A_0's Q-value = 0.15-0.3** after peaking
3. **Trust in A_0 = 0.8-0.9** after peaking

**Why the difference?**

Different update formulas and rates:
- Q-learning rate: 0.1 × 1.5 = 0.15 (exploratory)
- Trust learning rate: 0.03 (exploratory)

**But more importantly:**
- Q-value target: `scaled_reward` (range: -1 to +1)
- Trust target: `(scaled_reward + 1) / 2` (range: 0 to 1)

**For a +1.0 reward:**
- Q-value target: 1.0
- Trust target: 1.0

**For a 0.0 reward:**
- Q-value target: 0.0
- Trust target: 0.5

**For a -1.0 reward:**
- Q-value target: -1.0
- Trust target: 0.0

**The shift matters!** Trust has a higher baseline:
- Average reward 0.0 → Q converges to 0.0
- Average reward 0.0 → Trust converges to 0.5

This means trust values are systematically **higher** than Q-values!

### The Real Issue: Measurement vs Reality

**What's being measured (aggregated):**
- Average trust across all 5 AIs: **0.3-0.4** (diluted)

**What's actually happening:**
- Trust in primary AI (A_0): **0.8-0.9** (giant peak!)
- Trust in unused AIs: **0.2-0.3** (near initial)

**What tests show:**
- Tracking A_0 specifically: **0.8-0.9** (giant peak visible!)

### Why ALL Agent Types Show Same Pattern

**The user is right to be puzzled!** Both exploratory and exploitative agents show identical trust peaks because:

1. **Both use same Q-learning for action selection**
   - No alignment peeking (after Fix 0)
   - Both pick based on Q-values
   - Both converge on same "best" AI

2. **Both get positive feedback from the same AI**
   - If AI alignment is 0.9 (confirming), it confirms both agent types
   - If alignment is 0.1 (truthful), it's truthful to both
   - No agent-specific AI responses!

3. **Trust updates are agent-specific, but targets are not**
   - Exploratory: `trust_lr = 0.03`
   - Exploitative: `trust_lr = 0.015`
   - But target_trust is the same for both!
   - Exploratory just converges 2× faster

### The "Not Reflected in Q-Values" Mystery

User says trust peaks but Q-values don't. Let me check if this is a measurement issue too...

**Q-values collected in tests:**
Looking at test_dual_feedback.py lines 100-102:
```python
# Q-values
q_values_by_tick[agent_type]['A_0'].append(agent.q_table.get('A_0', 0.0))
q_values_by_tick[agent_type]['human'].append(agent.q_table.get('human', 0.0))
q_values_by_tick[agent_type]['self_action'].append(agent.q_table.get('self_action', 0.0))
```

**This DOES track A_0 specifically!** So Q-values should also show peaks.

**Unless...**

Let me check the Q-value update logic again:

```python
# Line 1330-1331
new_mode_q = old_mode_q + effective_learning_rate * (scaled_reward - old_mode_q)
```

For exploratory: effective_learning_rate = 0.15

**Simulation:** Starting Q = 0.0, receiving +1.0 rewards:
- Tick 1: Q = 0.0 + 0.15 × (1.0 - 0.0) = 0.15
- Tick 2: Q = 0.15 + 0.15 × (1.0 - 0.15) = 0.2775
- Tick 3: Q = 0.2775 + 0.15 × (1.0 - 0.2775) = 0.3859

After 10 +1.0 rewards: Q ≈ 0.8

**But:** Rewards aren't always +1.0! They're based on actual relief outcomes.

Looking at reward calculation (line 1284-1297):
```python
if self.agent_type == "exploratory":
    batch_reward = 0.8 * avg_actual_reward + 0.2 * (correct_ratio * 5.0)
else:
    batch_reward = 0.2 * avg_actual_reward + 0.8 * (correct_ratio * 5.0)
```

Then line 1310:
```python
scaled_reward = max(-1.0, min(1.0, batch_reward / 5.0))
```

So batch_reward of 5.0 → scaled_reward = 1.0

**Typical values:**
- avg_actual_reward: 0-2 (depends on disaster level)
- correct_ratio: 0-1 (proportion correct)
- batch_reward: 0-5
- scaled_reward: 0-1

**For exploratory agents with accurate sources:**
- avg_actual_reward ≈ 1.0
- correct_ratio ≈ 0.8
- batch_reward = 0.8 × 1.0 + 0.2 × 4.0 = 1.6
- scaled_reward = 0.32

**So Q-values peak around 0.3-0.4, not 0.8-0.9!**

**But trust?**
- target_trust = (0.32 + 1.0) / 2 = 0.66

Over multiple updates, trust can accumulate to 0.8+ while Q stays at 0.3-0.4.

## Summary: Why Trust Peaks But Q Doesn't

1. **Trust has higher baseline** due to (reward + 1)/2 transform
2. **Trust target for neutral reward (0.0) is 0.5, not 0.0**
3. **Typical scaled_reward is 0.2-0.4** (not 1.0)
4. **Q-values peak at 0.3-0.4** (tracking reward directly)
5. **Trust peaks at 0.7-0.9** (shifted by +0.5 baseline)

## Why This is Actually CORRECT Behavior

The trust transform is intentional:
- Q-values represent **expected future reward** (can be negative)
- Trust represents **confidence in source** (must be 0-1, centered at 0.5)

**The issue is:** Trust is accumulating too fast relative to evidence!

## The REAL Bug: Learning Rates Too High

With `trust_lr = 0.03` and typical rewards around 0.3:
- target_trust ≈ 0.65
- If current trust = 0.25:
- Δtrust = 0.03 × (0.65 - 0.25) = 0.012

**After just 10 updates:** trust goes from 0.25 → 0.4+
**After 30 updates:** trust approaches 0.65

**But agents only need 2-3 queries to start specializing on one AI!**

This creates runaway feedback:
1. Query A_0, get reward 0.3
2. A_0's Q = 0.0 → 0.045, trust = 0.25 → 0.26
3. Query A_0 again (highest Q), get reward 0.4
4. A_0's Q = 0.045 → 0.098, trust = 0.26 → 0.28
5. ...
6. After 10 queries: Q = 0.25, trust = 0.5
7. A_0 now dominates selection
8. Other AIs never get queried, stay at Q=0.0, trust=0.25
9. Trust in A_0 peaks at 0.8-0.9 while others stay low
10. **Average trust reported: 0.4** (hides the peak!)
11. **Test tracking A_0: 0.9** (shows the peak!)

## Recommended Fixes

### Fix A: Track trust in QUERIED AIs only
```python
# Instead of averaging all AIs:
ai_vals = [agent.trust[k] for k in agent.trust if k.startswith("A_")]

# Average only QUERIED AIs (minimum threshold):
ai_vals = [agent.trust[k] for k in agent.trust
           if k.startswith("A_") and agent.q_table.get(k, 0) != 0.0]
```

### Fix B: Report trust in PRIMARY AI
```python
# Find AI with highest Q-value (agent's preferred AI)
primary_ai = max([k for k in agent.q_table if k.startswith("A_")],
                 key=lambda k: agent.q_table[k])
primary_ai_trust = agent.trust[primary_ai]
```

### Fix C: Lower trust learning rates (as planned)
```python
explor_trust_lr = 0.006  # was 0.03 (divide by 5)
exploit_trust_lr = 0.003  # was 0.015 (divide by 5)
```

### Fix D: Rename metric to clarify
Instead of "AI trust", report:
- "Primary AI trust" (trust in most-queried AI)
- "Average AI trust" (current metric)
- "AI specialization" (variance in Q-values across AIs)
