# Baseline AI Factor - Impact Analysis

## Current Code (Lines 945-961)

### Exploratory Agent Biases:

**AI Bias Formula:**
```
ai_bias = (1.0 - alignment) * 0.5 + 0.1
         ^^^^^^^^^^^^^^^^^^^^^     ^^^
         inverse alignment     baseline_ai_factor
```

**Human Bias Formula:**
```
human_bias = alignment * 0.15
```

## Impact Across Alignment Levels

| Alignment | AI Bias | Human Bias | Net Preference |
|-----------|---------|------------|----------------|
| 0.0       | 0.60    | 0.00       | AI +0.60       |
| 0.25      | 0.475   | 0.0375     | AI +0.4375     |
| 0.50      | 0.35    | 0.075      | AI +0.275      |
| 0.75      | 0.225   | 0.1125     | AI +0.1125     |
| 1.0       | 0.10    | 0.15       | Human +0.05    |

## The Problem

### 1. **Weak Differentiation at High Alignment**
At alignment = 1.0, exploratory agents only slightly prefer humans (+0.05).
- This is too weak to drive clear behavioral differences
- Q-values and noise can easily overwhelm this small difference
- Result: Exploratory agents still use AI heavily at high alignment

### 2. **Distorts Experiment B's Core Hypothesis**
Experiment B tests: *"How does AI alignment affect information-seeking behavior?"*

**Expected behavior:**
- Low alignment → Exploratory prefer AI (truthful), Exploitative prefer humans
- High alignment → Exploratory prefer humans (diverse), Exploitative prefer AI (confirmation)

**Actual behavior with baseline:**
- Low alignment → Both types prefer AI (exploratory much more)
- High alignment → Both types still prefer AI (exploitative more, but exploratory also significantly)

### 3. **Explains Your Observed Issues**

#### Issue 1: "Exploratory agents use AI lots even with high alignment"
✓ **CONFIRMED** - At alignment=1.0, they still get +0.1 AI bias vs only +0.15 human bias

#### Issue 2: "SECI develops the same way for both agent types"
✓ **CONFIRMED** - Both types heavily use AI across all alignments, reducing behavioral differentiation

## What Happens WITHOUT the Baseline?

| Alignment | AI Bias (no baseline) | Human Bias | Net Preference |
|-----------|-----------------------|------------|----------------|
| 0.0       | 0.50                  | 0.00       | AI +0.50       |
| 0.25      | 0.375                 | 0.0375     | AI +0.3375     |
| 0.50      | 0.25                  | 0.075      | AI +0.175      |
| 0.75      | 0.125                 | 0.1125     | AI +0.0125     |
| 1.0       | 0.00                  | 0.15       | Human +0.15    |

### Benefits of Removing Baseline:

1. **Stronger differentiation** at high alignment (0.05 → 0.15 difference)
2. **Clear crossover point** around alignment = 0.75 where exploratory switch from AI to humans
3. **Clearer experimental results** - behavior changes more dramatically with alignment
4. **Better theoretical alignment** - exploratory agents actually avoid aligned AI

## Comparison: Exploitative vs Exploratory (WITHOUT baseline)

| Alignment | Exploitative AI Bias | Exploratory AI Bias | Behavioral Difference |
|-----------|---------------------|---------------------|----------------------|
| 0.0       | 0.0                 | 0.50                | Explor prefer AI +0.50 |
| 0.5       | 0.15                | 0.25                | Explor prefer AI +0.10 |
| 1.0       | 0.30                | 0.00                | Exploit prefer AI +0.30 |

**Clear behavioral flip:**
- Low alignment: Exploratory heavily prefer AI (truth-seeking)
- High alignment: Exploitative heavily prefer AI (confirmation-seeking)

## Recommendation

**Remove the baseline_ai_factor (line 946)** to:
- ✓ Restore intended experimental design
- ✓ Create clear behavioral differentiation
- ✓ Make results interpretable for publication
- ✓ Align behavior with theoretical predictions

**Alternative (if some baseline needed):**
Make it much smaller (0.02) or negative at high alignment.

## Why Was It Added?

Comment on line 946: "alignment more important than AI preference"

**Interpretation:** May have been added to ensure exploratory agents don't completely abandon AI at high alignment. But this:
- Contradicts the experimental hypothesis
- Prevents observing the natural behavioral dynamics
- Makes the alignment manipulation less effective
