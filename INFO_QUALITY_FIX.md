# INFO QUALITY FEEDBACK FIX

## Problem

Exploratory agents received ZERO information quality feedback events, preventing them from learning AI quality:

```
High Alignment:  Exploratory Info=0,   Exploitative Info=141
Low Alignment:   Exploratory Info=0,   Exploitative Info=78
```

## Root Cause

Information quality feedback requires agents to:
1. Query about cells → stored in `pending_info_evaluations`
2. **Observe those same cells** via `sense_environment()`
3. Compare reported vs observed values

**The Problem:**
- Exploratory agents query around `interest_point` (often far from position)
- `sense_environment()` only senses around **current position** (radius=2)
- **Never overlap** → Zero feedback events!

Exploitative agents query around `believed_epicenter` (near their area) so get some overlap.

## Solution: Prioritize Verifying Queried Cells

Implemented Option 1: Agents actively verify information they receive.

### Implementation (DisasterAI_Model.py)

**1. Track cells to verify** (Line 115-119):
```python
self.cells_to_verify = set()  # Cells queried about that need verification
```

**2. Add queried cells to priority list** (Line 1127-1130):
```python
# When receiving reports, add cells to verification priority
for cell, reported_value in reports.items():
    self.cells_to_verify.add(cell)
```

**3. Prioritize sensing these cells** (Line 339-360):
```python
def sense_environment(self):
    cells_to_sense = set()

    # 1. Priority cells (up to 5) - cells we queried about
    if self.cells_to_verify:
        priority_cells = list(self.cells_to_verify)[:5]
        for cell in priority_cells:
            cells_to_sense.add(cell)
            if random.random() < 0.3:  # Gradually remove
                self.cells_to_verify.discard(cell)

    # 2. Normal neighborhood (radius=2 from position)
    neighborhood = self.model.grid.get_neighborhood(pos, ...)
    cells_to_sense.update(neighborhood)

    # Process all cells
    for cell in cells_to_sense:
        # ... sense and evaluate ...
```

## Expected Results

With this fix, exploratory agents should:
1. Query about distant cells (interest_point)
2. **Actively verify** those cells over next few ticks
3. **Receive info quality feedback** comparing reported vs actual
4. **Learn AI quality** through 10x stronger feedback rewards
5. **Decrease trust** in confirming AI (high alignment)
6. **Increase trust** in truthful AI (low alignment)

This makes the model realistic: agents verify information they receive, enabling them to distinguish good from bad sources.

## Files Modified

- `DisasterAI_Model.py:115-119` - Added cells_to_verify tracking
- `DisasterAI_Model.py:1127-1130` - Add cells when querying
- `DisasterAI_Model.py:335-365` - Modified sense_environment() to prioritize verification

## Backup

Original code backed up in: `DisasterAI_Model_BACKUP_before_confirmation_fix.py`
