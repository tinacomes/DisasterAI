# Far-Away Cell Verification Analysis

## CURRENT SITUATION

### What Agents Query About

**Explorers**:
- Query about **uncertain cells** from `find_exploration_targets()`
- These are cells with: level >= 1, low confidence (high uncertainty)
- These cells come from agent's belief grid (all cells, anywhere on map)
- Query radius = 3 around the uncertain cell
- **Interest point can be FAR from agent's position**

**Example scenario**:
- Agent at position (5, 5), sensing radius = 2
- Agent has belief about cell (25, 25) with level=3, confidence=0.3 (uncertain)
- Agent queries about (25, 25) with radius=3
- Gets reports about cells (22-28, 22-28) - **NONE of which agent can sense!**

### Current Verification Mechanism

**For close-by cells** (within sensing radius = 2): ✓
- Agent queries about cell
- Agent directly senses cell (within 3-15 ticks)
- Compares reported level vs actual level
- Updates Q-values and trust based on accuracy
- **This works perfectly**

**For far-away cells** (outside sensing radius = 2): ✗
- Agent queries about cell
- Receives report from source
- Updates belief using Bayesian updating
- **BUT NEVER VERIFIES** because:
  - Agent doesn't sense the cell (too far away)
  - `pending_info_evaluations` never gets evaluated
  - No feedback on source accuracy for these cells!

### Only Current Verification for Far Cells

**Relief outcome feedback** (15-25 tick delay):
- If agent sends relief based on beliefs about far cells
- Later gets feedback on actual disaster levels
- Updates Q-values and trust
- **Problem**: Only works IF relief is sent to those specific cells

## THE VERIFICATION GAP

### Key Issue
Explorers specifically seek **uncertain cells** to query about. These are often:
- Far from current position
- Never directly sensed
- Never verified except through rare relief outcomes

### Consequence
- Explorers query many sources about far-away uncertain cells
- They update beliefs based on reports
- They **never learn which sources are accurate** about those cells
- Q-learning can't work properly for far-away information

### Numbers
With grid size 30x30 and sensing radius 2:
- Agent can sense ~12-20 cells (depending on position)
- Agent has beliefs about 900 cells (entire grid)
- **~98% of beliefs are about cells agent cannot directly verify!**

---

## SUGGESTED SOLUTION: TRUE TRIANGULATION

### Core Idea
For cells outside sensing range, verify by **comparing multiple sources** instead of comparing to direct sensing.

### Mechanism

#### Step 1: Track Multiple Reports Per Cell
```python
# In HumanAgent.__init__
self.cell_reports = {}  # {cell: [(tick, source_id, reported_level), ...]}
```

#### Step 2: When Receiving Report About Far Cell
```python
# In seek_information(), after getting report
distance_to_interest = distance(self.pos, interest_point)

if distance_to_interest > self.sensing_radius + 1:  # Far away cell
    # Track this report for triangulation
    for cell, reported_level in reports.items():
        if cell not in self.cell_reports:
            self.cell_reports[cell] = []
        self.cell_reports[cell].append((
            self.model.tick,
            source_id,
            reported_level
        ))
```

#### Step 3: Triangulation Verification (Periodic)
```python
def triangulate_far_cells(self):
    """
    Verify far-away cells by comparing multiple source reports.
    Sources that agree with consensus get trust boost.
    Outliers get trust penalty.
    """
    for cell, reports in self.cell_reports.items():
        # Need at least 2 reports to triangulate
        if len(reports) < 2:
            continue

        # Only triangulate recent reports (within 20 ticks)
        recent_reports = [
            (tick, src, lvl) for tick, src, lvl in reports
            if self.model.tick - tick <= 20
        ]

        if len(recent_reports) < 2:
            continue

        # Calculate consensus level (median or mode)
        levels = [lvl for _, _, lvl in recent_reports]
        consensus_level = calculate_consensus(levels)  # median or mode

        # Calculate variance/disagreement
        variance = calculate_variance(levels)

        # Evaluate each source against consensus
        for tick, source_id, reported_level in recent_reports:
            error = abs(reported_level - consensus_level)

            # Reward for agreement, penalty for deviation
            if error == 0:
                triangulation_reward = 0.3 * (1.0 / (variance + 1))
            elif error == 1:
                triangulation_reward = 0.1 * (1.0 / (variance + 1))
            elif error == 2:
                triangulation_reward = -0.2
            else:
                triangulation_reward = -0.5

            # Update Q-values and trust
            self.update_from_triangulation(source_id, triangulation_reward)

        # Clear old reports for this cell
        self.cell_reports[cell] = [
            r for r in reports
            if self.model.tick - r[0] <= 20
        ]
```

### Benefits

1. **Explorers can verify far-away uncertain cells**
   - Query multiple sources about same uncertain cell
   - Compare reports to find consensus
   - Learn which sources are reliable for far information

2. **Exploiters also benefit**
   - Can verify their believed epicenter through multiple confirmations
   - But they prefer friends (so less diversity in sources)

3. **Realistic disaster information seeking**
   - In real disasters, can't directly verify all reports
   - Must cross-check with multiple sources
   - Consensus-building mimics real emergency response

4. **Fast feedback**
   - Don't wait for relief outcome (15-25 ticks)
   - Can verify as soon as 2+ sources queried (3-10 ticks)
   - Faster Q-learning

### Implementation Complexity

**Medium** - requires:
- Track multiple reports per cell
- Consensus calculation logic
- Periodic triangulation evaluation (every 10 ticks?)
- Memory management (clean old reports)

---

## ALTERNATIVE: MOVEMENT-BASED VERIFICATION

### Core Idea
Agents occasionally move toward their most uncertain cells to verify directly.

### Mechanism
```python
# Explorers: every N ticks, move toward most uncertain cell
if self.model.tick % 15 == 0 and self.agent_type == "exploratory":
    if self.exploration_targets:
        target = self.exploration_targets[0]  # Most uncertain cell
        # Move toward target (simple pathfinding)
        dx = sign(target[0] - self.pos[0])
        dy = sign(target[1] - self.pos[1])
        new_pos = (self.pos[0] + dx, self.pos[1] + dy)
        if valid_position(new_pos):
            self.model.grid.move_agent(self, new_pos)
```

### Benefits
- Simpler implementation
- Eventually verifies uncertain cells through sensing
- Realistic behavior (explorers investigate)

### Drawbacks
- Slow (takes many ticks to reach far cells)
- Agents might cluster in one area
- Doesn't verify all queried cells (only targets)

---

## ALTERNATIVE: NEIGHBOR CONSISTENCY

### Core Idea
Verify far cells by checking consistency with neighboring cells that CAN be sensed.

### Mechanism
```python
def verify_by_neighbor_consistency(self, far_cell, reported_level):
    """
    Check if reported level is consistent with nearby cells we CAN sense.
    Disasters have spatial correlation - neighbors should have similar levels.
    """
    # Find neighbors of far_cell that we CAN sense
    sensable_neighbors = [
        n for n in get_neighbors(far_cell, radius=2)
        if distance(self.pos, n) <= self.sensing_radius
    ]

    if not sensable_neighbors:
        return None  # Can't verify this way

    # Get actual levels of sensable neighbors
    neighbor_levels = [
        self.model.disaster_grid[n[0], n[1]]
        for n in sensable_neighbors
    ]

    # Expected far_cell level based on neighbors (with distance decay)
    expected_level = estimate_from_neighbors(neighbor_levels, distances)

    # Compare reported vs expected
    error = abs(reported_level - expected_level)

    # Fuzzy verification (not perfect, but some signal)
    if error <= 1:
        return 0.2  # Somewhat consistent
    elif error == 2:
        return -0.1  # Somewhat inconsistent
    else:
        return -0.3  # Very inconsistent
```

### Benefits
- Uses existing sensing capability
- No need to track multiple reports
- Leverages spatial correlation of disasters

### Drawbacks
- Only works if some neighbors are sensable
- Fuzzy/imprecise (disasters aren't perfectly smooth)
- Might not work well for disaster edges

---

## RECOMMENDATION

**Primary: TRIANGULATION** (compare multiple sources)
- Most realistic for far-away cells
- Enables true multi-source verification
- Directly addresses "triangulation" concept
- Medium complexity but high value

**Secondary: MOVEMENT-BASED** (as backup/complement)
- Explorers occasionally move toward uncertain cells
- Provides ground-truth verification eventually
- Simple to implement

**Combined approach**:
1. Use triangulation for initial fast feedback (2-10 ticks)
2. Use movement to eventually get ground truth (20-50 ticks)
3. Use neighbor consistency as fallback when applicable

This gives explorers multiple verification pathways and makes "exploration" more meaningful.

---

## QUESTIONS FOR USER

1. **Which verification approach do you prefer?**
   - Triangulation (compare multiple sources)?
   - Movement-based (explorers move to verify)?
   - Neighbor consistency (spatial correlation)?
   - Combination?

2. **How often should triangulation happen?**
   - Every tick (expensive)?
   - Every 10 ticks (periodic)?
   - Only when agent has idle time?

3. **Minimum sources for triangulation?**
   - 2 sources (minimal)?
   - 3 sources (more robust)?
   - Dynamic (more sources for higher confidence)?

4. **Should explorers move toward uncertain cells?**
   - Always (deterministic)?
   - Sometimes (probabilistic)?
   - Never (pure query-based)?

5. **Relief outcome feedback sufficient?**
   - Keep as-is (15-25 tick delay)?
   - Or must add faster verification for far cells?
