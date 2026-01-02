# Trial Version - Baseline Fix

## Changes Made to DisasterAI_Model.py

### Problem Identified
The `baseline_ai_factor = 0.1` prevented exploratory agents from switching away from AI at high alignment, distorting the experimental design.

### Solution Implemented

**Lines 938-962: Exploratory Agent Source Selection Logic**

#### BEFORE (Distorted):
```python
inverse_alignment_factor = (1.0 - alignment) * 0.5
baseline_ai_factor = 0.1  # DISTORTION!
ai_bias = inverse_alignment_factor + baseline_ai_factor
human_bias = alignment * 0.15
```

#### AFTER (Corrected):
```python
inverse_alignment_factor = (1.0 - alignment) * 0.5
ai_bias = inverse_alignment_factor  # Pure inverse alignment
human_bias = alignment * 0.25  # Strengthened from 0.15
```

## Expected Behavioral Changes

### Exploratory Agents - NEW Behavior

| Alignment | AI Bias | Human Bias | Net Preference |
|-----------|---------|------------|----------------|
| 0.0       | +0.50   | 0.00       | AI +0.50       |
| 0.25      | +0.375  | +0.0625    | AI +0.3125     |
| 0.50      | +0.25   | +0.125     | AI +0.125      |
| 0.75      | +0.125  | +0.1875    | Human +0.0625  |
| 1.0       | 0.00    | +0.25      | Human +0.25    |

**Crossover point:** ~0.67 alignment (exploratory switch from AI to humans)

### Comparison: Exploitative vs Exploratory

| Alignment | Exploitative → AI | Exploratory → AI | Differentiation |
|-----------|-------------------|------------------|-----------------|
| 0.0       | 0.0               | +0.50            | Explor prefer AI |
| 0.5       | +0.15             | +0.25            | Explor prefer AI |
| 0.75      | +0.225            | +0.125           | Exploit prefer AI |
| 1.0       | +0.30             | 0.00             | Exploit prefer AI |

## Theoretical Alignment

### Exploitative Agents (Confirmation-Seeking)
- **Low alignment:** AI is truthful → not useful for confirmation → prefer friends/self
- **High alignment:** AI confirms beliefs → very useful → PREFER AI ✓

### Exploratory Agents (Correctness-Seeking)
- **Low alignment:** AI is truthful → very useful for accurate info → PREFER AI ✓
- **High alignment:** AI is biased → not useful for correctness → PREFER HUMANS ✓

## Impact on Metrics

### SECI (Social Echo Chamber Index)
**Before:** Similar patterns for both agent types (both using AI heavily)
**After:** Should see clear divergence:
- Exploitative: More negative SECI at high alignment (AI-driven echo chambers)
- Exploratory: Less negative SECI at high alignment (human network diversity)

### AECI (AI Echo Chamber Index)
**Before:** Both types show high AECI across alignments
**After:** Should see clear divergence:
- Exploitative: AECI increases with alignment
- Exploratory: AECI decreases with alignment (crossover ~0.67)

## Next Steps

1. **Re-run Experiment B** with corrected logic
2. **Verify behavioral differentiation** in the results
3. **Compare** trial results to original distorted results
4. **If successful:** This becomes the publication version

## Files Modified
- `DisasterAI_Model.py` (lines 938-962)
