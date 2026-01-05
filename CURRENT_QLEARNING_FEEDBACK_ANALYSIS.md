# Current Q-Learning Feedback Mechanism for Explorers

## SCENARIO: Explorer Queries Far-Away Uncertain Cell

### Step-by-Step Current Flow

**Tick 10**: Explorer at position (5, 5)
- Has belief about cell (25, 25): level=3, confidence=0.3 (UNCERTAIN)
- `find_exploration_targets()` identifies (25, 25) as target
- Queries source "A_0" about (25, 25) with radius=3
- Gets report: "level=5" for cell (25, 25)

**Tick 10**: `update_belief_bayesian()` called (line 724)
- Updates belief: (25, 25) now level=4, confidence=0.5 (Bayesian blend)
- Adds to `pending_info_evaluations` (line 838):
  ```python
  self.pending_info_evaluations.append((
      tick=10,
      source_id="A_0",
      cell=(25, 25),
      reported_level=5
  ))
  ```

**Tick 11-25**: Agent continues operating
- Agent is at position (5, 5) or nearby (if moved)
- Sensing radius = 2
- Distance to (25, 25) = sqrt((25-5)^2 + (25-5)^2) = sqrt(800) ≈ 28
- **Cell (25, 25) is OUTSIDE sensing range!**

**Tick 25**: `evaluate_information_quality()` called during sensing (line 395)
- Checks `pending_info_evaluations` for cell (25, 25)
- Evaluation window: 3-15 ticks (line 408-417)
- Tick 25 is 15 ticks after tick 10 → **WINDOW EXPIRED**
- Item removed from pending list (line 412)
- **NO Q-LEARNING FEEDBACK GIVEN!**

### Result: ZERO FEEDBACK FOR FAR CELLS

**Explorer queried about far-away cell but got:**
- ❌ No info quality feedback (never sensed the cell)
- ❌ No Q-value update for source "A_0"
- ❌ No trust update for source "A_0"
- ✓ Only feedback: IF relief is sent to (25, 25), get outcome 15-25 ticks later

---

## CURRENT FEEDBACK PATHWAYS

### Pathway 1: Info Quality Feedback (Fast, 3-15 ticks)
**Location**: `evaluate_information_quality()` line 395-487

**How it works**:
1. Agent queries source about cell → adds to `pending_info_evaluations`
2. Agent later senses the same cell (within 3-15 ticks)
3. Compares reported_level vs actual_level
4. Updates Q-values and trust based on accuracy

**Q-learning update** (line 443-458):
```python
# Calculate accuracy reward
if level_error == 0: accuracy_reward = 0.5
elif level_error == 1: accuracy_reward = 0.2
elif level_error == 2: accuracy_reward = -0.3
else: accuracy_reward = -0.7

# Update mode Q-value (e.g., "ai" or "human")
old_mode_q = self.q_table[mode]
info_learning_rate = 0.25  # For explorers
new_mode_q = old_mode_q + info_learning_rate * (accuracy_reward - old_mode_q)
self.q_table[mode] = new_mode_q

# Update specific source Q-value (e.g., "A_0")
old_q = self.q_table[source_id]
new_q = old_q + info_learning_rate * (accuracy_reward - old_q)
self.q_table[source_id] = new_q

# Update trust
trust_target = 0.5 + 0.5 * accuracy_reward  # Maps -0.7→0.15, 0→0.5, +0.5→0.75
trust_change = 0.15 * (trust_target - old_trust)  # For explorers
self.trust[source_id] = old_trust + trust_change
```

**Works for**: Close-by cells (within sensing radius)
**Fails for**: Far-away cells (never sensed)

---

### Pathway 2: Relief Outcome Feedback (Slow, 15-25 ticks)
**Location**: `process_reward()` line 1244-1412

**How it works**:
1. Agent sends relief to cells based on beliefs
2. 15-25 ticks later, learns actual disaster levels
3. Calculates reward based on accuracy (0.8) + confirmation (0.2) for explorers
4. Updates Q-values and trust for the source that provided info

**Q-learning update** (line 1387-1405):
```python
# Calculate batch reward (accuracy + confirmation blend)
scaled_reward = batch_reward / 5.0  # Normalize to [-1, 1]

# Update mode Q-value
old_mode_q = self.q_table[mode]
effective_learning_rate = 0.15 * 1.5  # 0.225 for explorers
new_mode_q = old_mode_q + effective_learning_rate * (scaled_reward - old_mode_q)
self.q_table[mode] = new_mode_q

# Update specific source Q-value
old_q = self.q_table[source_id]
new_q = old_q + effective_learning_rate * (scaled_reward - old_q)
self.q_table[source_id] = new_q

# Update trust
target_trust = (scaled_reward + 1.0) / 2.0  # Map to [0,1]
trust_change = trust_learning_rate * (target_trust - old_trust)
self.trust[source_id] = old_trust + trust_change
```

**Works for**: Any cell where relief was sent
**Problem**: Explorers don't send relief to ALL uncertain cells they query!

---

## THE FEEDBACK GAP FOR EXPLORERS

### Typical Explorer Behavior
1. Queries about 5-10 uncertain cells per tick (seeking information)
2. Sends relief to only top 3-5 highest-believed cells per tick
3. **Most queried uncertain cells DON'T get relief**

### Coverage Analysis

**Assume grid 30x30 = 900 cells**

**Per tick, Explorer**:
- Queries: ~1 source about ~1-3 uncertain cells (query radius=3 → ~7-20 cells reported)
- Senses: ~12-20 cells (sensing radius=2)
- Sends relief: ~3-5 cells (highest believed level)

**Feedback coverage**:
- Info quality feedback: Only for cells that are BOTH queried AND sensed
  - With sensing radius=2, very few far cells are ever sensed
  - **Coverage: ~5-10% of queried cells**
- Relief outcome feedback: Only for cells that get relief
  - **Coverage: ~30-50% of queried cells IF explorer sends relief to uncertain cells**
  - **But explorers may prefer high-confidence cells for relief!**

**Result**:
- **50-70% of queries get ZERO feedback!**
- Explorers cannot learn which sources are accurate about far uncertain cells
- Q-learning is severely hobbled

---

## WHY TRIANGULATION IS CRITICAL

### Without Triangulation
Explorer queries "A_0" about far cell (25, 25):
- Gets report: "level=5"
- Updates belief to level=4 (Bayesian blend)
- Never verifies → **no feedback**
- Next time, still has Q["A_0"]=0.0 (unchanged)
- **Random source selection continues indefinitely**

### With Triangulation
Explorer queries multiple sources about same far cell:
- Tick 10: Query "A_0" → reports "level=5"
- Tick 12: Query "H_3" → reports "level=4"
- Tick 14: Query "A_1" → reports "level=5"

**Triangulation (tick 15)**:
- Consensus: level=5 (two sources agree)
- "A_0": error=0 → reward=+0.3 → Q["A_0"] increases
- "H_3": error=1 → reward=+0.1 → Q["H_3"] increases slightly
- "A_1": error=0 → reward=+0.3 → Q["A_1"] increases

**Next query**: Explorer prefers "A_0" or "A_1" (higher Q-values)
**Q-learning works!**

---

## SIMPLIFIED TRIANGULATION PROPOSAL

### Lightweight Implementation
Instead of full triangulation system, use **opportunistic comparison**:

**When querying about a cell that already has recent info**:
```python
# In update_belief_bayesian(), after receiving report
if cell in self.recent_reports:  # Track last 2-3 reports per cell
    previous_reports = self.recent_reports[cell]

    # If we have 1+ previous reports within 20 ticks
    if len(previous_reports) >= 1:
        # Simple consensus: average of all reports
        all_reports = [r[1] for r in previous_reports] + [reported_level]
        consensus = round(sum(all_reports) / len(all_reports))

        # Evaluate current source against consensus
        error = abs(reported_level - consensus)

        # Quick reward
        if error == 0: reward = 0.3
        elif error == 1: reward = 0.1
        else: reward = -0.2

        # Update Q-value and trust IMMEDIATELY
        update_q_and_trust(source_id, reward)

        # Also re-evaluate previous sources
        for prev_tick, prev_level, prev_source in previous_reports:
            prev_error = abs(prev_level - consensus)
            if prev_error == 0: prev_reward = 0.3
            elif prev_error == 1: prev_reward = 0.1
            else: prev_reward = -0.2
            update_q_and_trust(prev_source, prev_reward)

# Track this report for future comparisons
self.recent_reports[cell].append((
    self.model.tick,
    reported_level,
    source_id
))
```

### Benefits
- **No separate triangulation loop** - happens during normal belief updates
- **Immediate feedback** - as soon as 2nd report received
- **Covers far cells** - works for any cell, sensed or not
- **Simple consensus** - just average (can improve later)
- **Re-evaluates old reports** - updates Q-values retrospectively

### Complexity
**Low** - ~30-50 lines of code, no new data structures except `self.recent_reports`

---

## COMPARISON: CURRENT vs SIMPLIFIED TRIANGULATION

| Aspect | Current | With Simplified Triangulation |
|--------|---------|-------------------------------|
| Feedback for close cells | ✓ Info quality (3-15 ticks) | ✓ Both mechanisms |
| Feedback for far cells | ❌ Only if relief sent (15-25 ticks) | ✓ Triangulation (immediate) |
| Coverage | ~30-50% of queries | ~80-90% of queries |
| Learning speed | Slow (15-25 tick delay) | Fast (immediate) |
| Complexity | Current | +30-50 lines |

---

## RECOMMENDATION

**IMPLEMENT SIMPLIFIED TRIANGULATION**

### Why?
1. **Critical for explorers**: They specifically query far uncertain cells
2. **Fast feedback**: Immediate when 2nd report received
3. **Low complexity**: Simple addition to existing code
4. **High impact**: Enables Q-learning for 50-70% of queries that currently get no feedback
5. **Realistic**: Mimics real disaster info verification (cross-checking sources)

### Implementation Plan
1. Add `self.recent_reports = {}` to HumanAgent.__init__
2. In `update_belief_bayesian()`, check for previous reports
3. If found, calculate consensus and evaluate sources
4. Track current report for future comparisons
5. Clean old reports (>20 ticks) periodically

### Would this address your concerns?
- ✓ Provides Q-learning feedback for far cells
- ✓ No movement required (stays compatible with multiple high-priority cells)
- ✓ Works with existing exploration targeting
- ✓ Fast enough to influence learning during simulation

**Shall I implement this?**
