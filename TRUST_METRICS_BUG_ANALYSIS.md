# Trust Metrics Bug Analysis

## The Problem

User reports:
1. Trust peaks suddenly for ALL agent types (not averaged out)
2. This peak is NOT reflected in Q-values
3. Both exploratory and exploitative agents show identical peaking behavior

## Investigation: Trust Calculation Code

### How Trust is Stored (Per Agent)
```python
# In HumanAgent.__init__ (around line 111)
self.trust = {}
# Later populated with:
# - 'A_0', 'A_1', 'A_2', 'A_3', 'A_4' (5 AI agents)
# - 'H_5', 'H_12', etc. (specific human agents)
```

### How Trust is Updated (Line 1347-1356)
```python
# When relief outcome is processed:
if source_id in self.trust:
    old_trust = self.trust[source_id]

    # target_trust from line 1311: (scaled_reward + 1.0) / 2.0
    # If reward is positive → target_trust = 0.5 to 1.0
    # If reward is negative → target_trust = 0.0 to 0.5

    trust_change = self.trust_learning_rate * (target_trust - old_trust)
    new_trust = max(0.0, min(1.0, old_trust + trust_change))
    self.trust[source_id] = new_trust  # Updates SPECIFIC source (e.g., A_0)
```

### How Trust is COLLECTED for Metrics (Lines 2844-2847)
```python
# In DisasterModel.step(), every tick:
for agent in self.humans.values():
    # Collect ALL AI sources and AVERAGE them
    ai_vals = [agent.trust[k] for k in agent.trust if k.startswith("A_")]
    ai_mean = np.mean(ai_vals) if ai_vals else 0
```

## 🔴 BUG #1: Trust Collection Averaging Dilution

**The Issue:**
- There are 5 AI agents: A_0, A_1, A_2, A_3, A_4
- Each human agent has separate trust values for each AI
- But we **average across ALL 5 AI agents** when collecting metrics
- **If agents primarily query ONE AI (e.g., A_0), that trust will spike**
- **But the reported metric averages A_0 + A_1 + A_2 + A_3 + A_4**
- **Most A_i stay at initial value (0.25), diluting the signal**

**Example:**
```python
Agent trust dict:
{
    'A_0': 0.95,  # Heavily queried, trust spiked
    'A_1': 0.25,  # Never queried, stays at initial
    'A_2': 0.25,  # Never queried, stays at initial
    'A_3': 0.25,  # Never queried, stays at initial
    'A_4': 0.25,  # Never queried, stays at initial
}

# What gets reported in trust_stats:
ai_mean = (0.95 + 0.25 + 0.25 + 0.25 + 0.25) / 5 = 0.39  # DILUTED!
```

**But user says they SEE a giant peak, so this can't be the only issue...**

## 🔴 BUG #2: Which AI Gets Queried?

Let me check how AI agents are selected...

```python
# Line 1062-1068 (seek_information)
else:  # AI
    if chosen_mode in self.model.ais:
        source_id = chosen_mode
    else:
        # Otherwise, pick best AI for this query
        source_id = self.choose_best_ai_for_query(interest_point, query_radius)
```

**Key question:** What is `chosen_mode` when seeking AI?

Looking at lines 964-989 (action selection):
```python
# For AI sources, available_modes will contain individual AI IDs
for ai_agent in self.model.ai_list:
    ai_id = ai_agent.unique_id
    if ai_id not in available_modes:
        available_modes.append(ai_id)

# Then Q-table is checked:
scores = {}
for mode in available_modes:
    base_q = self.q_table.get(mode, 0.0)
    scores[mode] = base_q
```

So:
- Q-table has separate entries for A_0, A_1, A_2, A_3, A_4
- Action selection chooses SPECIFIC AI based on Q-values
- If A_0 has highest Q-value, it gets selected more

**This means agents CAN specialize on one AI!**

## 🔴 BUG #3: Q-values vs Trust Mismatch

**How are Q-values and Trust different?**

1. **Q-values** (line 1330-1345):
   - Updated from relief outcome (15-25 tick delay)
   - Drive action selection (which source to query)
   - Updated with learning_rate (0.1) × 1.5 for exploratory
   - Represents expected future reward

2. **Trust** (line 1347-1356):
   - Updated from relief outcome (same 15-25 tick delay)
   - Does NOT drive action selection for AI (Q-values do!)
   - Updated with trust_learning_rate (0.015-0.03)
   - Used only for selecting which HUMAN friend to query

**CRITICAL FINDING:**
- For AI queries: Q-values drive selection, NOT trust
- For human queries: Trust drives selection (line 1047-1049)
- Trust in AI is being tracked but **not used for anything important**!

## 🔴 BUG #4: Why Do We See Giant Peaks?

If trust collection averages across 5 AIs and dilutes the signal, why does user see giant peaks?

**Hypothesis 1: All agents query the same AI**
- If ALL agents converge on A_0 (highest Q-value)
- Then everyone's A_0 trust spikes
- But A_1-A_4 stay at 0.25
- Averaging gives: (0.95 + 0.25×4)/5 = 0.39

Still diluted! User wouldn't see "giant peak".

**Hypothesis 2: Trust metrics are tracked differently in tests**
Looking at test_dual_feedback.py line 105:
```python
# Trust in AI
trust_by_tick[agent_type].append(agent.trust.get('A_0', 0.5))
```

**BINGO! The test code tracks ONLY A_0!**

So in test code:
- Tracking: `agent.trust.get('A_0', 0.5)` → shows giant peak
- In model metrics: `np.mean(all_ai_trusts)` → diluted

**But if trust doesn't drive AI selection, why do we care about peaks?**

## 🔴 BUG #5: Q-values SHOULD show peaks too

User says: "Trust peaks but NOT reflected in Q-values"

Let me trace this:
- Line 1311: `target_trust = (scaled_reward + 1.0) / 2.0`
- Line 1310: `scaled_reward = max(-1.0, min(1.0, batch_reward / 5.0))`
- Line 1330-1331: Q-value update uses the SAME scaled_reward

```python
# Q-value update (line 1330-1331)
new_mode_q = old_mode_q + effective_learning_rate * (scaled_reward - old_mode_q)

# Trust update (line 1353)
trust_change = self.trust_learning_rate * (target_trust - old_trust)
```

**Key difference:**
- Q-value: Uses `scaled_reward` directly (range: -1 to +1)
- Trust: Uses `target_trust = (scaled_reward + 1)/2` (range: 0 to 1)

**And learning rates:**
- Q-learning rate: 0.1 × 1.5 = 0.15 (exploratory)
- Trust learning rate: 0.03 (exploratory)

**So Q-values should update 5× FASTER than trust!**

If trust peaks, Q-values should peak EVEN MORE.

## The Real Bug: Update Formula Difference

Let's simulate:

**Scenario: Agent gets one good reward (+1.0)**

Initial values:
- Q-value: 0.0 (initialization for new sources)
- Trust: 0.25 (initial_ai_trust)

After one +1.0 reward:
- Q-value: `0.0 + 0.15 * (1.0 - 0.0) = 0.15`
- Trust: `0.25 + 0.03 * (1.0 - 0.25) = 0.25 + 0.0225 = 0.2725`

After second +1.0 reward:
- Q-value: `0.15 + 0.15 * (1.0 - 0.15) = 0.2775`
- Trust: `0.2725 + 0.03 * (1.0 - 0.2725) = 0.294125`

After 10 consecutive +1.0 rewards:
- Q-value: converges toward 1.0 (exponentially)
- Trust: converges toward 1.0 (exponentially)

**But Q converges ~5× faster!**

## Wait... Actual Bug?

Let me re-check the update formulas:

**Q-value update (TD-learning):**
```python
new_q = old_q + learning_rate * (reward - old_q)
```
This is: `new_q = old_q + α(r - old_q)`

**Trust update:**
```python
trust_change = trust_lr * (target_trust - old_trust)
new_trust = old_trust + trust_change
```
This is: `new_trust = old_trust + α(target - old_trust)`

**SAME FORMULA! Just different α values.**

So with α_Q = 0.15 and α_trust = 0.03:
- Q-values update 5× faster than trust
- Both converge exponentially to target

## 🚨 THE ACTUAL BUG 🚨

**Q-values and Trust track DIFFERENT SOURCE IDs!**

Looking more carefully at the code:

```python
# Line 1324-1331: Update MODE Q-value
if mode in self.q_table:
    old_mode_q = self.q_table[mode]
    new_mode_q = old_mode_q + effective_learning_rate * (scaled_reward - old_mode_q)
    self.q_table[mode] = new_mode_q

# Line 1334-1345: Update SPECIFIC source Q-values
for source_id in source_ids:
    if source_id in self.q_table:
        old_q = self.q_table[source_id]
        new_q = old_q + effective_learning_rate * (scaled_reward - old_q)
        self.q_table[source_id] = new_q
```

**Both mode AND source_id Q-values are updated!**

But for AI queries:
- `mode` = "A_0" (or A_1, A_2, etc.)
- `source_ids` = ["A_0"]
- So mode == source_id for AI!

**Not a mismatch for AI queries.**

## HYPOTHESIS: The Giant Peak is REAL

Let me reconsider. If user sees giant peaks in trust:

1. Tests track `agent.trust['A_0']` specifically
2. Model collects average across all AIs
3. If most agents never query most AIs, most AI trusts stay at 0.25
4. But A_0 (most queried) spikes to 0.8+
5. Average: 0.39 (diluted)

**But user says they see giant peaks in the data!**

This means:
- Either tests are showing A_0 specifically (which peaks)
- Or there's aggregation happening that I'm missing

Let me check if there's a visualization issue...

## Check: What Gets Plotted?

In plot_trust_evolution (line 5326):
```python
data_slice = trust_stats_array[:, :, data_index]
```

Trust stats structure (line 2880):
```python
self.trust_stats.append((tick, ai_exp_mean, friend_exp_mean, nonfriend_exp_mean,
                        ai_expl_mean, friend_expl_mean, nonfriend_expl_mean))
```

So:
- Index 0: tick
- Index 1: ai_exp_mean (exploitative agents' average AI trust)
- Index 4: ai_expl_mean (exploratory agents' average AI trust)

And ai_exp_mean is computed as (line 2856):
```python
ai_exp_mean = np.mean([x[0] for x in trust_exp])
```

Where trust_exp contains (line 2851):
```python
trust_exp.append((ai_mean, friend_mean, nonfriend_mean))
```

And ai_mean is (line 2847):
```python
ai_mean = np.mean(ai_vals) if ai_vals else 0
```

Where ai_vals is (line 2844):
```python
ai_vals = [agent.trust[k] for k in agent.trust if k.startswith("A_")]
```

**CONFIRMED: It averages across ALL 5 AI agents.**

So if there's a peak showing up in the data, it means:
- MOST AI agents are being queried (not just A_0)
- OR agents cycle through AIs rapidly
- OR the initial trust values are being counted incorrectly

## 🔍 SMOKING GUN: Initial Trust Values

Check line 2844:
```python
ai_vals = [agent.trust[k] for k in agent.trust if k.startswith("A_")]
```

**Question: What if some agents don't have all 5 AI agents in their trust dict yet?**

Let me check how trust dict is initialized...
