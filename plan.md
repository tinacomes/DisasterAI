# Comprehensive Fix Plan

Five interdependent fixes that together make AI alignment experiments meaningful.

---

## Fix 1: Scarcity Sensing with Explore/Exploit Observation

**Problem**: Agents sense 25 cells/tick → rumor zone overwritten in ~5 ticks → AI alignment irrelevant.

**Solution**: 2 cells/tick (own cell + 1 strategic pick from radius-2).

### Changes in `sense_environment()` (lines 521-591):

**Replace** the full-neighborhood loop with:

```python
def sense_environment(self):
    pos = self.pos
    radius = 2

    # Tier 1: Always sense own cell
    self._sense_cell(pos)

    # Tier 2: 1 strategic cell from neighborhood
    cells = self.model.grid.get_neighborhood(pos, moore=True, radius=radius, include_center=False)
    valid = [c for c in cells if 0 <= c[0] < self.model.width and 0 <= c[1] < self.model.height]

    if valid:
        if self.agent_type == "exploitative":
            target = self._pick_exploit_observation(valid)
        else:
            target = self._pick_explore_observation(valid)
        self._sense_cell(target)

    self.flush_belief_rewards()
```

### New `_sense_cell(cell)` method (extract from lines 526-586):

Same noise model, confidence assignment, belief blending — just refactored into a callable helper. **No behavioral change** to per-cell logic.

### New `_pick_exploit_observation(valid_cells)`:

```python
def _pick_exploit_observation(self, valid_cells):
    """Pick cell in vicinity closest to believed epicenter."""
    self.find_believed_epicenter()
    if self.believed_epicenter:
        return min(valid_cells,
                   key=lambda c: (c[0]-self.believed_epicenter[0])**2 +
                                 (c[1]-self.believed_epicenter[1])**2)
    # Fallback: highest-level believed cell
    return max(valid_cells,
               key=lambda c: self.beliefs.get(c, {}).get('level', 0))
```

### New `_pick_explore_observation(valid_cells)`:

```python
def _pick_explore_observation(self, valid_cells):
    """Pick cell in vicinity with highest uncertainty."""
    def uncertainty(c):
        b = self.beliefs.get(c)
        if not b or not isinstance(b, dict):
            return 1.0
        return 1.0 - b.get('confidence', 0.1)
    top_k = sorted(valid_cells, key=uncertainty, reverse=True)[:3]
    return random.choice(top_k)
```

### What stays the same:
- `is_within_sensing_range()` — still defines potential range
- All downstream systems (queries, evaluation, relief, metrics)

---

## Fix 2: Triangulation via Memory System

**Problem**: Single observation → 0.85-0.98 confidence. Beliefs become unshakeable immediately. This also causes the **overshoot** issue.

**Solution**: Route direct sensing through the memory system. Confidence then depends on multiple agreeing observations, not single reads.

### Changes in `_sense_cell()`:

Instead of setting `self.beliefs[cell]` directly, call `add_to_memory()`:

```python
def _sense_cell(self, cell):
    # ... noise model produces belief_level, belief_conf (as observation weight) ...

    # Route through memory — confidence determined by convergence, not single reads
    self.add_to_memory(cell, belief_level, belief_conf, source_id=self.unique_id)

    # Still call these for Q-learning feedback
    self.reward_belief_accuracy(cell, actual)
    self.evaluate_self_action(cell, actual)
```

### Lower per-observation confidence (fixes overshoot):

These become **observation weights** feeding into memory, not direct belief confidence:

```python
# Exploitative
belief_conf = 0.4                    # was 0.6
if belief_level >= 3: belief_conf = 0.5   # was 0.85
elif belief_level > 0: belief_conf = 0.45  # was 0.75

# Exploratory
belief_conf = 0.5                    # was 0.9
if belief_level > 0: belief_conf = 0.55    # was 0.98
```

### How confidence builds (via existing `_update_belief_from_memory`):

The memory system already computes belief confidence from:
- **Memory fullness** (30%): `len(memory) / memory_size`
- **Agreement** (70%): `1.0 - (std(levels) / 2.5)`

With memory_size=3 (exploiter) or 7 (explorer):
- 1st observation: fullness=0.33, agreement=0.5 → confidence ≈ 0.45
- 2nd agreeing observation: fullness=0.67, agreement=1.0 → confidence ≈ 0.90
- 2nd **disagreeing** observation: fullness=0.67, agreement~0.3 → confidence ≈ 0.41

This means:
- Rumors (confidence 0.6) survive 1 contradicting observation
- 2+ agreeing observations needed to build high confidence
- AI reports and human reports count as memory entries too (already use add_to_memory)
- All sources contribute to the same convergence requirement

### Remove direct belief writes from sense_environment:

Delete lines 556-581 (the blending logic) and lines 579-581 (new belief creation). These are replaced by the memory system path.

### What stays the same:
- `_update_belief_from_memory()` — unchanged, already does exactly what we need
- `add_to_memory()` — unchanged, already handles FIFO and source metadata

---

## Fix 3: Strengthen Disaster Dynamics

**Problem**: `disaster_dynamics=2` changes ~1 cell per tick (10% chance). Over 150 ticks, ~15 cells change out of 900. Old observations stay valid almost forever.

**Solution**: Make multiple cells change per tick, with both increases AND decreases, spatially correlated near the disaster zone.

### Rewrite `update_disaster()` (lines 3000-3048):

```python
def update_disaster(self):
    if self.disaster_grid is not None:
        self.previous_grid = self.disaster_grid.copy()

    if self.disaster_dynamics == 0:
        pass  # Static

    elif self.disaster_dynamics == 1:
        # Slow: 1-2 cells, small changes
        n_changes = random.randint(1, 2)
        for _ in range(n_changes):
            if random.random() < 0.3:
                self._evolve_cell(magnitude_range=(1, 2))

    elif self.disaster_dynamics == 2:
        # Medium: 3-5 cells per tick, mix of growth and recovery
        n_changes = random.randint(3, 5)
        for _ in range(n_changes):
            self._evolve_cell(magnitude_range=(1, 3))

    elif self.disaster_dynamics == 3:
        # Rapid: 5-8 cells per tick, large swings
        n_changes = random.randint(5, 8)
        for _ in range(n_changes):
            self._evolve_cell(magnitude_range=(2, 4))

    # Detect significant changes (existing logic)
    if self.previous_grid is not None:
        grid_change = np.abs(self.disaster_grid - self.previous_grid)
        max_change = np.max(grid_change)
        if max_change >= self.event_threshold:
            self.event_ticks.append(self.tick)
```

### New `_evolve_cell()` helper:

```python
def _evolve_cell(self, magnitude_range=(1, 3)):
    """Evolve a single cell — spatially weighted toward disaster zone, both up and down."""
    # 70% chance: change near epicenter (realistic aftershocks/recovery)
    # 30% chance: random location (new developments)
    if random.random() < 0.7:
        # Near epicenter: offset by gaussian
        sigma = self.disaster_radius * 0.8
        x = int(np.clip(np.random.normal(self.epicenter[0], sigma), 0, self.width - 1))
        y = int(np.clip(np.random.normal(self.epicenter[1], sigma), 0, self.height - 1))
    else:
        x = random.randint(0, self.width - 1)
        y = random.randint(0, self.height - 1)

    magnitude = random.randint(*magnitude_range)
    current = self.disaster_grid[x, y]

    # 60% intensify, 40% recover (net growth but with churn)
    if random.random() < 0.6:
        self.disaster_grid[x, y] = min(5, current + magnitude)
    else:
        self.disaster_grid[x, y] = max(0, current - magnitude)
```

### Impact:
- `dynamics=2`: 3-5 cells change per tick → ~600 cell-changes over 150 ticks
- Old observations go stale; agents need continuous re-sensing
- Supports triangulation: single old observation ≠ current ground truth
- Spatially correlated changes make epicenter beliefs drift realistically

---

## Fix 4: Fix `evaluate_pending_info` Dead Code

**Problem**: Multiple dead code paths, inconsistencies with `evaluate_information_quality`, broken explorer logic for remote cells.

### Fix 4a: Remove unused `use_prior_as_reference` flag (line 1079)

The flag is set but never read. The reference reassignment on lines 1090-1091 already does the work.

```python
# DELETE line 1079: use_prior_as_reference = False
# DELETE line 1089: use_prior_as_reference = True
# KEEP lines 1090-1091 (the actual reference reassignment)
```

### Fix 4b: Replace broken explorer remote-cell logic (lines 1113-1211)

**Current problem**: Two contradictory approaches in the same method:
1. Lines 1113-1138: Ground truth check via `is_within_sensing_range()` — contradicts `is_remote_cell` definition (dead code)
2. Lines 1195-1208: Confidence-improvement metric — circular (uses belief that may have been set by the report itself), rewards can exceed bounds (up to 1.48)

**Replace with**: Match `evaluate_information_quality`'s 5-case explorer logic, using stored prior as reference when ground truth unavailable:

```python
if self.agent_type == "exploitative":
    # Unchanged: 95% confirmation, 5% accuracy
    combined_reward = 0.95 * confirmation_score + 0.05 * accuracy_score
else:  # EXPLORATORY
    # Match evaluate_information_quality logic (lines 922-949)
    prior_conf_stored = stored_prior_conf if stored_prior_conf else 0.5
    belief_changed = (prior_error >= 1)
    was_uncertain = (prior_conf_stored < 0.5)

    if is_remote_cell:
        # Remote: can't verify accuracy → use accuracy_score=0.0 (neutral)
        info_was_accurate_proxy = (accuracy_score >= 0.5)  # best guess from reference
    else:
        info_was_accurate_proxy = (level_error <= 1)

    if belief_changed and info_was_accurate_proxy:
        combined_reward = 0.8
    elif belief_changed and not info_was_accurate_proxy:
        combined_reward = -0.4
    elif not belief_changed and was_uncertain and info_was_accurate_proxy:
        combined_reward = 0.5
    elif not belief_changed and not was_uncertain and info_was_accurate_proxy:
        combined_reward = 0.0
    else:
        combined_reward = -0.3

    # Confidence increase bonus (matching evaluate_information_quality line 944-949)
    current_belief = self.beliefs.get(cell, {})
    current_conf = current_belief.get('confidence', 0.5) if isinstance(current_belief, dict) else 0.5
    if current_conf - prior_conf_stored > 0.15:
        combined_reward += 0.25
```

### Fix 4c: Remove extra `confidence_scaling` from Q and trust updates (lines 1228, 1235, 1251)

`evaluate_information_quality` uses `source_knowledge_conf` only. `evaluate_pending_info` adds an extra `confidence_scaling` multiplier. Align them:

```python
# Line 1228: was base_lr * confidence_scaling * source_knowledge_conf
info_lr = base_lr * source_knowledge_conf  # match evaluate_information_quality

# Line 1235: same fix
info_lr = base_lr * source_knowledge_conf

# Line 1251: was base_trust_lr * confidence_scaling * source_knowledge_conf
trust_lr = base_trust_lr * source_knowledge_conf  # match evaluate_information_quality
```

Also remove the `confidence_scaling` variable (line 1106) — no longer used.

### Fix 4d: Remove unreachable `else: mode = None` (lines 1220-1221)

Source IDs always start with "A_" or "H_". The else branch is dead code. Remove it and the `if mode and` guard on line 1224 (mode is always set).

---

## Fix 5: Interactions and Consistency Checks

These aren't separate fixes but verification that the 4 fixes work together:

### 5a: `initialize_beliefs()` — no change needed
Rumors are injected into `self.beliefs` at startup (lines 151-220). With triangulation, these initial beliefs will be in `self.beliefs` but NOT in `self.belief_memory`. The first direct sensing of a rumor cell will add to memory, and `_update_belief_from_memory` will compute confidence from memory contents — naturally lower than the rumor confidence, requiring corroboration. **This is correct behavior.**

### 5b: `evaluate_self_action()` — no change needed
Called per-cell in `_sense_cell()`. Now processes 2 cells/tick instead of 25. Already handles variable queue sizes.

### 5c: `reward_belief_accuracy()` + `flush_belief_rewards()` — no change needed
Batch averages over 2 cells instead of 25. Less noise, more meaningful signal per sample.

### 5d: `apply_confidence_decay()` — no change needed
Distance-based decay still meaningful: within-range cells CAN be sensed, outside-range cells can't. The scarcity creates natural staleness; decay handles the rest.

### 5e: `report_beliefs()` — no change needed
When agents report beliefs to each other, they report from `self.beliefs` which is now memory-derived. Lower base confidence → reports carry lower confidence → recipients need more corroboration. **Cascading benefit.**

---

## Summary Table

| # | Fix | Lines | Effort | Risk |
|---|-----|-------|--------|------|
| 1 | Scarcity sensing (2 cells/tick) | 521-591 | Rewrite + 3 new methods | Low — isolated to sense_environment |
| 2 | Triangulation via memory | Inside _sense_cell | Small — route through existing memory | Low — memory system already works |
| 3 | Strengthen dynamics | 3000-3048 | Rewrite + 1 helper | Low — isolated to update_disaster |
| 4 | Fix evaluate_pending_info | 1079-1251 | Edit ~40 lines | Medium — reward logic, test carefully |
| 5 | Verify interactions | N/A | Read-only | None |

## Implementation Order

1 → 2 (depend on each other — _sense_cell is created in Fix 1, modified in Fix 2)
3 (independent)
4 (independent)
5 (verification after all fixes)
