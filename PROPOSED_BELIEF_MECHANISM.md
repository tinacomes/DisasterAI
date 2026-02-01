# Proposed Belief Update Mechanism

## Problem Summary

### Issue 1: Source Precision Ignores Source-Specific Information
Current implementation only uses receiver's trust in source. Missing:
- Source's confidence in their own report
- Whether source directly sensed or guessed
- Source's agent type (exploiter might exaggerate, explorer might be more accurate)
- Source's historical accuracy

### Issue 2: Bayesian vs Memory
Bayesian assumes perfect rationality. Memory-based is more cognitively realistic.

---

## Proposed Solution: Memory-Based with Rich Source Information

### 1. Enhanced Report Structure

Change `report_beliefs()` to return metadata:

```python
def report_beliefs(self, interest_point, query_radius):
    """Reports human agent's beliefs WITH metadata."""
    report = {}

    cells_to_report_on = self.model.grid.get_neighborhood(
        interest_point, moore=True, radius=query_radius, include_center=True
    )

    for cell in cells_to_report_on:
        if 0 <= cell[0] < self.model.width and 0 <= cell[1] < self.model.height:
            if cell in self.beliefs and isinstance(self.beliefs[cell], dict):
                belief_info = self.beliefs[cell]

                # Return RICH information
                report[cell] = {
                    'level': belief_info.get('level', 0),
                    'confidence': belief_info.get('confidence', 0.1),
                    'source_type': self.agent_type,  # 'exploitative' or 'exploratory'
                    'is_sensed': self.is_within_sensing_range(cell),  # Did they directly sense it?
                    'last_updated': belief_info.get('last_updated', 0),  # How recent?
                }
            else:
                # Guessing - lower confidence
                guessed_value = self._guess_cell_value(cell)
                if guessed_value is not None:
                    report[cell] = {
                        'level': guessed_value,
                        'confidence': 0.2,  # Low confidence for guesses
                        'source_type': self.agent_type,
                        'is_sensed': False,
                        'is_guess': True,
                    }

    return report
```

### 2. Memory-Based Belief Storage

Replace single belief per cell with memory of recent info-bits:

```python
class HumanAgent:
    def __init__(self, ...):
        # ... existing init ...

        # Memory-based belief storage
        self.belief_memory = {}  # {cell: [(level, confidence, source_id, tick), ...]}
        self.memory_size = 5  # Remember last 5 reports per cell

        # Derived beliefs (computed from memory)
        self.beliefs = {}  # {cell: {'level': int, 'confidence': float}}

    def add_to_memory(self, cell, level, confidence, source_id):
        """Add info-bit to memory, forget oldest if full."""
        if cell not in self.belief_memory:
            self.belief_memory[cell] = []

        # Add new info with timestamp
        self.belief_memory[cell].append({
            'level': level,
            'confidence': confidence,
            'source_id': source_id,
            'tick': self.model.tick,
            'source_trust': self.trust.get(source_id, 0.1),
        })

        # Enforce memory limit (FIFO - forget oldest)
        if len(self.belief_memory[cell]) > self.memory_size:
            self.belief_memory[cell].pop(0)

        # Update derived belief
        self._update_belief_from_memory(cell)

    def _update_belief_from_memory(self, cell):
        """Compute belief as weighted average of memory."""
        if cell not in self.belief_memory or not self.belief_memory[cell]:
            return

        memory = self.belief_memory[cell]

        # Option A: Simple mean (paper's approach)
        # levels = [m['level'] for m in memory]
        # new_level = int(round(np.mean(levels)))

        # Option B: Trust-weighted mean (hybrid approach)
        total_weight = 0
        weighted_sum = 0
        for m in memory:
            # Weight by source trust AND source confidence AND recency
            recency = 1.0 - (self.model.tick - m['tick']) / 100.0  # Decay over 100 ticks
            recency = max(0.1, recency)

            weight = m['source_trust'] * m['confidence'] * recency
            weighted_sum += m['level'] * weight
            total_weight += weight

        if total_weight > 0:
            new_level = int(round(weighted_sum / total_weight))
        else:
            new_level = int(round(np.mean([m['level'] for m in memory])))

        # Confidence based on memory fullness and agreement
        memory_fullness = len(memory) / self.memory_size
        levels = [m['level'] for m in memory]
        agreement = 1.0 - (np.std(levels) / 2.5) if len(levels) > 1 else 0.5
        new_confidence = 0.3 * memory_fullness + 0.7 * agreement

        self.beliefs[cell] = {
            'level': max(0, min(5, new_level)),
            'confidence': max(0.1, min(0.95, new_confidence)),
        }
```

### 3. Source-Specific Precision Calculation

When receiving info, calculate precision based on ALL available information:

```python
def calculate_source_precision(self, report_metadata, source_id):
    """
    Calculate precision based on source-specific factors.

    Factors:
    1. Receiver's trust in source (learned over time)
    2. Source's confidence in their report
    3. Whether source directly sensed vs guessed
    4. Source's agent type (explorers typically more accurate)
    5. Recency of source's information
    """
    base_trust = self.trust.get(source_id, 0.1)

    # Factor 1: Source's own confidence
    source_confidence = report_metadata.get('confidence', 0.5)

    # Factor 2: Sensed vs guessed
    is_sensed = report_metadata.get('is_sensed', False)
    sensing_multiplier = 1.5 if is_sensed else 0.7

    # Factor 3: Source agent type
    source_type = report_metadata.get('source_type', 'unknown')
    if source_type == 'exploratory':
        type_multiplier = 1.2  # Explorers tend to be more accurate
    elif source_type == 'exploitative':
        type_multiplier = 0.9  # Exploiters might confirm biases
    else:
        type_multiplier = 1.0

    # Factor 4: Is this a friend? (social trust boost)
    is_friend = source_id in self.friends
    friend_multiplier = 1.3 if is_friend and self.agent_type == "exploitative" else 1.0

    # Combined precision
    raw_precision = (
        base_trust *
        source_confidence *
        sensing_multiplier *
        type_multiplier *
        friend_multiplier
    )

    # Convert to precision scale (0 to ~10)
    precision = 5.0 * raw_precision / (1 - raw_precision + 0.1)

    # Cap based on receiver's agent type
    if self.agent_type == "exploitative":
        precision = min(precision, 6.0)  # More skeptical
    else:
        precision = min(precision, 10.0)  # More accepting

    return precision
```

### 4. Updated Belief Integration

Combine D/δ acceptance with memory-based storage:

```python
def integrate_information(self, cell, report_metadata, source_id):
    """
    Full information integration pipeline:
    1. Calculate acceptance probability (D/δ formula)
    2. If accepted, add to memory
    3. Update derived belief from memory
    """
    reported_level = report_metadata['level']

    # Get current belief (from memory average)
    current_belief = self.beliefs.get(cell, {'level': 0, 'confidence': 0.1})
    prior_level = current_belief['level']

    # ===== STEP 1: Acceptance Decision (D/δ formula) =====
    d = abs(reported_level - prior_level)
    d_normalized = d / 5.0

    if d_normalized > 0:
        D_normalized = self.D / 5.0
        delta = self.delta

        # Paper formula
        p_accept = (D_normalized ** delta) / (d_normalized ** delta + D_normalized ** delta)

        # Modify by source precision
        source_precision = self.calculate_source_precision(report_metadata, source_id)
        precision_modifier = min(1.5, 0.5 + source_precision / 10.0)
        p_accept *= precision_modifier

        # Friend modifier for exploiters
        if self.agent_type == "exploitative" and source_id in self.friends:
            p_accept = min(1.0, p_accept * 1.3)

        if random.random() >= p_accept:
            # REJECT - don't add to memory
            return False

    # ===== STEP 2: Add to Memory =====
    self.add_to_memory(
        cell,
        reported_level,
        report_metadata.get('confidence', 0.5),
        source_id
    )

    # Memory update automatically recalculates self.beliefs[cell]
    return True
```

---

## Comparison: Three Approaches

| Aspect | Current (Bayesian) | Paper (Memory Mean) | Proposed (Memory + Rich) |
|--------|-------------------|---------------------|--------------------------|
| Acceptance | Hardcoded thresholds | D/δ formula | D/δ + source precision |
| Update | Precision-weighted | Simple mean of memory | Trust-weighted mean of memory |
| Source info | Only trust | None | Trust + confidence + type + sensed |
| Memory | No (single belief) | Yes (N info-bits) | Yes (N info-bits with metadata) |
| Forgetting | Never | FIFO (oldest first) | FIFO with recency weighting |
| Realism | Low (perfect Bayes) | Medium (bounded) | High (bounded + source factors) |

---

## Recommended Parameter Values

### Memory Parameters
| Parameter | Exploiter | Explorer | Rationale |
|-----------|-----------|----------|-----------|
| `memory_size` | 3 | 7 | Exploiters filter more, remember less |
| `recency_weight` | 0.3 | 0.5 | Explorers value recent info more |

### D/δ Parameters (unchanged)
| Parameter | Exploiter | Explorer |
|-----------|-----------|----------|
| D | 1.5 | 3.0 |
| δ | 20 | 8 |

### Source Precision Multipliers
| Factor | Multiplier Range |
|--------|------------------|
| Sensed (vs guessed) | 1.5 / 0.7 |
| Explorer source | 1.2 |
| Exploiter source | 0.9 |
| Friend (for exploiters) | 1.3 |

---

## Implementation Priority

1. **Phase 1**: Add metadata to `report_beliefs()` (both Human and AI)
2. **Phase 2**: Implement memory storage with FIFO
3. **Phase 3**: Implement source-specific precision calculation
4. **Phase 4**: Integrate D/δ acceptance formula
5. **Phase 5**: Calibrate memory_size and weighting parameters
