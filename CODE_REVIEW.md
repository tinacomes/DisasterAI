# Comprehensive Code Review: DisasterAI Simulation

**Reviewer:** Claude Code
**Date:** 2026-02-01
**Status:** Complete

---

## Executive Summary

The codebase implements a disaster response simulation with human agents (exploiters/explorers), AI agents, social networks, and Q-learning for source prioritization. This review identifies what is **implemented correctly**, what is **questionable or inconsistent**, and what is **missing or problematic**.

---

## 1. DISASTER ENVIRONMENT (Lines 2331-2343)

**Status: IMPLEMENTED CORRECTLY**

- Grid created with Gaussian decay around random epicenter
- `disaster_radius` derived from `share_of_disaster` parameter
- Levels range 0-5, with epicenter at 5
- Dynamic evolution via `update_disaster()` with `disaster_dynamics` parameter (0=static, 1-3=increasing evolution speed)

**Note:** No "Experiment D" file exists - the `disaster_dynamics` parameter is the mechanism, but there's no dedicated experiment file testing it.

---

## 2. SOCIAL NETWORKS (Lines 2643-2777)

**Status: IMPLEMENTED CORRECTLY**

- Creates 2-3 connected components (communities)
- Dense within-community connections (70% probability)
- No between-community connections (ensures separate echo chambers)
- Type diversity ensured within communities (both exploiters/explorers in each)

---

## 3. RUMOR MECHANISM (Lines 2382-2426)

**Status: IMPLEMENTED CORRECTLY**

- Per-component rumor assignment (same rumor to all agents in a social network component)
- Rumor epicenter kept separate from actual disaster (minimum separation distance)
- Rumor details: epicenter location, intensity, confidence, radius
- Applied during `initialize_beliefs()` (Lines 146-230)

---

## 4. AGENT TYPES (Lines 48-2004)

**Status: IMPLEMENTED with some inconsistencies**

### Exploiters (correctly implemented):
- Query believed epicenter (confirmation-seeking) - Line 1362-1416
- Higher D threshold (2.0) = reject conflicting info more
- Higher prior precision (2.5x) = resist belief changes
- Reward sources that confirm beliefs (0.95 * confirmation_score) - Line 833
- Slow trust learning rate (0.015-0.03)
- Friends get rejection probability reduction (Line 1193-1194)

### Explorers (correctly implemented):
- Query high-uncertainty areas - Line 1417-1431 via `find_highest_uncertainty_area()`
- Lower D threshold (4.0) = accept more info
- Lower prior precision (0.6x) = more open to change
- Reward sources for accuracy (0.95 * accuracy_score) - Line 846
- Faster trust learning rate (0.03-0.06)

### Issues Found:

1. **D/delta VALUES - VERIFY INTENTION**: At Line 124-125:
   ```python
   self.D = 2.0 if agent_type == "exploitative" else 4
   self.delta = 3.5 if agent_type == "exploitative" else 1.2
   ```
   - D=2.0 for exploiters means they only accept when level difference < 2
   - D=4.0 for explorers means they accept when level difference < 4
   - This appears correct (exploiters reject more), but delta seems unused in the rejection logic

2. **Sensing radius is SAME for both types** (radius=2 at Line 443). If explorers should have larger sensing radius, this is NOT implemented.

---

## 5. AI AGENT (Lines 2006-2235)

**Status: IMPLEMENTED CORRECTLY**

- Larger sensing: 15% of grid per tick (Line 2014)
- Memory system for temporal consistency (Line 2026-2027)
- Alignment mechanism works correctly:
  - Low alignment (0.1) = 1% noise probability = truthful
  - High alignment (0.9) = 9% noise probability = confirming

- `report_beliefs()` applies alignment bias (Lines 2202-2229):
  - Calculates difference between caller beliefs and AI sensed values
  - Adjusts report toward caller beliefs based on alignment level
  - Higher alignment = more confirmation of caller's existing beliefs

---

## 6. BELIEF UPDATE MECHANISM (Lines 1162-1337)

**Status: PARTIALLY MATCHES THE PAPER**

The implementation uses a **Bayesian precision-weighting approach**:

```python
posterior_precision = prior_precision + source_precision
posterior_level = (prior_precision * prior_level + source_precision * reported_level) / posterior_precision
posterior_confidence = posterior_precision / (1 + posterior_precision)
```

This is a **legitimate Bayesian update** where:
- Prior precision = agent's confidence in existing belief
- Source precision = source's trustworthiness
- Update is weighted by relative precisions

The referenced paper (British Journal of Social Psychology) typically describes updates in terms of opinion change as a function of discrepancy. The current implementation captures this via:
- D/delta parameters controlling acceptance probability (Lines 1179-1203)
- Rejection mechanism for exploiters when `level_diff` is high

---

## 7. Q-LEARNING (Lines 1462-1524, 1703-1875)

**Status: FIXED but COMPLEX**

### Three-Level Structure:
1. **Mode Selection**: "self_action", "human", "ai" (epsilon-greedy)
2. **Source Selection**: Within mode, pick specific source
3. **Q-Value Updates**: Both mode and specific source Q-values updated

### Key Fixes Already Applied (per FIXES_SUMMARY.md):
- Issue #0 (CRITICAL): Removed "alignment cheating" - agents no longer peek at `ai_alignment_level`
- Issue #1: Now updates BOTH mode Q-value AND specific source Q-value
- Issue #2: Removed alignment-based trust logic
- Issue #3: Extended evaluation window (3-15 ticks for normal, 30 for explorer remote cells)

### Reward Calculation:
- **Exploiters**: 80% confirmation score + 20% actual reward (Lines 1800-1805)
- **Explorers**: 80% actual accuracy + 20% correctness ratio (Lines 1795-1800)

### Potential Issue - MODE BIASES:
```python
# Lines 1488-1507
if self.agent_type == "exploitative":
    scores["human"] += 0.1   # exploit_friend_bias
    scores["self_action"] += 0.1  # exploit_self_bias
else:  # exploratory
    scores["human"] += 0.2
    scores["ai"] += 0.2
    scores["self_action"] -= 0.1
```
The +0.2 biases for explorers toward human AND ai may distort Q-learning signal - agents start biased rather than learning purely from experience. Consider whether this is intended.

---

## 8. RELIEF & FEEDBACK (Lines 1639-1875)

**Status: IMPLEMENTED CORRECTLY**

### Relief Mechanism (Line 1639-1700):
- Targets cells with belief level >= 3, weighted by score
- Exploiters weight by confidence^1.5
- Explorers weight 70% level + 30% exploration (1-confidence)
- Queues rewards with 15-25 tick delay

### Dual-Timeline Feedback:
1. **Info Quality Feedback** (3-15 ticks): `evaluate_information_quality()` - Lines 511-679
   - Evaluates whether reported info matched ground truth
   - Exploiters: 80% confirmation + 20% accuracy reward
   - Explorers: 80% accuracy + 20% confirmation reward

2. **Relief Outcome Feedback** (15-25 ticks): `process_reward()` - Lines 1703-1875
   - Evaluates whether targeted cells actually needed relief
   - Updates Q-values and trust based on outcomes

### Key Fix Applied:
- Explorer remote cell evaluation deferred until sensed (Line 843: `continue`)
- Extended expiry window (30 ticks) for explorer remote cells (Line 724)

---

## 9. SECI/AECI METRICS (Lines 3039-3337)

**Status: IMPLEMENTED CORRECTLY**

### SECI (Social Echo Chamber Index):
```python
seci_val = (friend_var - global_var) / global_var  # for negative (echo chamber)
seci_val = (friend_var - global_var) / (max_possible_var - global_var)  # for positive
```
- Range: [-1, +1]
- Negative = friends less diverse than global (echo chamber)
- Positive = friends more diverse than global (anti-echo chamber)

### AECI (AI Echo Chamber Index):
- Same variance-based methodology
- Compares AI-reliant agents' belief variance vs global variance
- Separate tracking for exploiters vs explorers

### Additional Metrics:
- Belief Error (MAE)
- Belief Variance
- Trust Statistics
- Component-level SECI/AECI
- Information Diversity (Shannon Entropy)
- Retainment metrics (acceptance ratios)

---

## 10. CRITICAL ISSUES IDENTIFIED

### Issue A: Agent Movement Not Implemented
Agents have positions but no movement logic. Line 2376-2378 places agents initially, but there's no `move()` call in `step()`. Agents appear **static**, which affects:
- Sensing (always same cells)
- Exploration targets (may be unreachable)
- Info quality feedback (agents can't "move to verify")

### Issue B: Experiment D Not Found
The description mentions "disaster environment... dynamically evolving (driven by parameters tested in experiment D)" - no `test_experiment_d.py` or similar exists. The `disaster_dynamics` parameter exists but isn't tested in a dedicated experiment.

### Issue C: 5-Tick Delay Discrepancy
**Description mentions**: "with a delay of 5 ticks"
**Code implements**: Info feedback uses 3-15 tick window; Relief feedback uses 15-25 tick delay.

The 5-tick delay isn't directly implemented.

### Issue D: Legacy/Dead Code
Many older model files exist but aren't used:
- `DisBeliefMod.py`, `DisMod.py`, `Model2.py`, `DisasterModel.py`, `DisasterModel2.py`
- `DetectionAction.py`
- `DisasterBubbles.py` (has visualization functions that may duplicate main model)

### Issue E: Triangulation Not Explicit
The code mentions triangulation but there's no explicit triangulation logic beyond cross-reference in `evaluate_pending_info()`.

---

## 11. WHAT MATCHES INTENDED DESIGN

| Component | Status | Notes |
|-----------|--------|-------|
| Disaster grid with Gaussian | CORRECT | Lines 2331-2343 |
| Dynamic disaster evolution | CORRECT | `disaster_dynamics` parameter |
| Social network components | CORRECT | Multiple communities |
| Rumor mechanism | CORRECT | Per-component assignment |
| Exploiter behavior | CORRECT | Confirmation-seeking, reject conflicting |
| Explorer behavior | CORRECT | Uncertainty-seeking, accuracy-focused |
| AI alignment mechanism | CORRECT | Truthful (low) vs confirming (high) |
| Bayesian belief updates | CORRECT | Precision-weighted |
| Q-learning (3-mode) | CORRECT | After fixes |
| Dual-timeline feedback | CORRECT | Info (3-15) + Relief (15-25) |
| SECI/AECI metrics | CORRECT | Variance-based |

---

## 12. WHAT DOESN'T MATCH OR IS MISSING

| Component | Status | Issue |
|-----------|--------|-------|
| Agent movement | MISSING | Agents are static |
| Experiment D | MISSING | No dedicated experiment file |
| 5-tick delay | DIFFERENT | Uses 3-15 / 15-25 tick windows |
| Triangulation | UNCLEAR | Not explicitly implemented |
| Explorer larger sensing | NOT IMPLEMENTED | Both types use radius=2 |

---

## 13. RECOMMENDATIONS

1. **Add agent movement** if explorers are supposed to physically explore

2. **Create Experiment D** for disaster dynamics testing

3. **Consider removing initial biases** from explorer Q-learning to ensure pure learning

4. **Clean up legacy files** - archive or remove unused model versions

5. **Document the actual feedback delays** vs the intended 5-tick delay

6. **Verify D/delta values** match intended exploiter/explorer behavior

7. **Add explicit triangulation** if intended as separate mechanism

---

## 14. FILE ORGANIZATION

### Primary Files:
- `DisasterAI_Model.py` - Main simulation (~8,153 lines)
- `test_dual_feedback.py` - Experiment A
- `test_filter_bubbles.py` - Experiment B
- `test_agent_improvements.py` - Multi-agent tests

### Documentation:
- `EXPERIMENTAL_DESIGN.md` - Primary reference
- `FIXES_SUMMARY.md` - Q-learning bug fixes
- `ISSUES_ANALYSIS.md` - Issue analysis

### Legacy (consider archiving):
- `DisBeliefMod.py`, `DisMod.py`, `Model2.py`
- `DisasterModel.py`, `DisasterModel2.py`
- `DetectionAction.py`, `DisasterBubbles.py`

---

## 15. CONCLUSION

The core simulation mechanics are **correctly implemented**. The major gaps are:
1. Missing agent movement
2. Missing Experiment D
3. Different feedback delays than described

The Q-learning fixes documented in `FIXES_SUMMARY.md` have been applied correctly. The dual-timeline feedback mechanism is sophisticated and well-implemented, with appropriate fixes for explorer confirmation bias.

The codebase would benefit from cleanup of legacy files and documentation updates to match actual implementation details.
