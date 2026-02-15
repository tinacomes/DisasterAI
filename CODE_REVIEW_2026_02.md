# DisasterAI Code Review - February 2026

## Executive Summary

The model runs without crashes. The core architecture is sound: D/delta acceptance
via `integrate_information` is active and properly routes to memory-based belief
storage. However, 7 significant issues affect whether the simulation produces the
expected behavioral differentiation between explorer/exploiter agents across
alignment conditions.

---

## Findings

### 1. CRITICAL: Two Parallel Belief Update Pipelines

**Active path:** `integrate_information()` (line 737) → `add_to_memory()` → `_update_belief_from_memory()`
- Uses D/delta acceptance formula from Geschke et al. 2019
- Memory-based weighted averaging of info-bits
- Called from `seek_information()` at line 2006

**Dead path:** `update_belief_bayesian()` (line 1534)
- Uses hardcoded rejection thresholds (lines 1552-1575)
- Bayesian precision-weighted updates
- **Never called during simulation**

`update_belief_bayesian` has its own rejection logic, its own pending_info tracking,
and its own trust/confidence updates. If accidentally re-enabled, it would silently
switch the entire belief mechanism. BELIEF_UPDATE_ANALYSIS.md still describes this
dead path as "current implementation."

**Action:** Delete `update_belief_bayesian()` or mark deprecated. Update docs.

---

### 2. CRITICAL: Explorer AI Trust/Usage Doesn't Differentiate by Alignment

Diagnostic results (150 ticks, 3 seeds averaged):

| Metric                 | Alignment=0.1 | Alignment=0.9 |
|------------------------|---------------|---------------|
| Explorer Q(ai)         | 0.078         | 0.051         |
| Explorer AI trust      | 0.495         | 0.498         |
| Explorer AI usage      | 52.2%         | 53.2%         |

Expected: explorers should strongly prefer truthful AI and penalize confirming AI.
Observed: near-zero differentiation.

**Root causes:**
1. Trust decay (0.012/tick toward 0.5) nearly cancels learned trust changes
2. Remote cell evaluation gives neutral accuracy scores for explorers (line 1134-1138)
3. Q-value reward scaling pushes values into a narrow range near zero

**Action:** Reduce explorer trust decay, improve remote-cell feedback, or increase
feedback learning rates.

---

### 3. SIGNIFICANT: Double-Dipping in Information Feedback

Two methods process the same `pending_info_evaluations` list:
- `evaluate_information_quality()` - called per-cell during `sense_environment()`
- `evaluate_pending_info()` - called once per tick in `step()`

They use **different reward scaling:**
- `evaluate_information_quality`: `accuracy_reward = combined_reward * 0.5`
- `evaluate_pending_info`: `accuracy_reward = combined_reward * 0.7 - 0.1`

Same event type gets inconsistent reward depending on when/how it's evaluated.

**Action:** Unify reward scaling or consolidate into one evaluation path.

---

### 4. SIGNIFICANT: Exploiter Q-values Differentiate but Behavior Doesn't

Exploiter Q(ai): 0.284 (alignment=0.1) vs 0.488 (alignment=0.9) — correct direction.
But AI usage: 29.4% vs 30.9% — negligible change.

**Root cause:** Epsilon=0.3 means 30% random exploration. With 3 modes, random alone
gives ~33% AI usage. Q-value differences can't overcome epsilon + biases + noise.

**Action:** Add epsilon decay, or increase Q-value feedback magnitude.

---

### 5. SIGNIFICANT: SECI Sign Convention Inconsistency

- `DisasterModel.step()`: SECI = (friend_var - global_var) / global_var → **negative = echo chamber**
- `validate_social_network()`: SECI = (global_var - component_var) / global_var → **positive = echo chamber**
- `calculate_component_seci()`: Same as validate — inverted from step()

**Action:** Standardize. Recommend negative = echo chamber throughout.

---

### 6. MODERATE: Three Independent Trust Update Channels

Trust is updated in 4-5 places per query event:
1. `evaluate_information_quality()` — ground truth comparison
2. `evaluate_pending_info()` — cross-reference comparison
3. `update_trust_for_accuracy()` — direct belief-vs-reality adjustment
4. `process_reward()` — relief outcome feedback
5. `apply_trust_decay()` — every tick

This makes calibration extremely difficult. The net effect of any single feedback
event is unpredictable when 3-4 channels modify the same trust value.

**Action:** Remove `update_trust_for_accuracy()` (it's a redundant channel).
Consolidate info-quality feedback into one path.

---

### 7. MODERATE: AI Alignment is Continuous Blending, Not Probability

At alignment=0.9 with human_conf=0.7, trust=0.3:
`alignment_factor = 0.9 * (1 + 1.4) + 0.9 * 0.3 * 0.7 ≈ 2.35`

The AI overshoots — pushes values 2.35x past human beliefs. Even alignment=0.1
biases reports 26% toward human beliefs. Only alignment=0 gives pure truth.

**Action:** Document this behavior. Consider whether alignment should be a probability
of aligning vs telling truth, rather than a continuous multiplier.

---

## What Works Correctly

- D/delta acceptance formula is properly implemented and active
- Memory-based belief storage with FIFO works as designed
- Disaster grid initialization and dynamic evolution work
- Social network creates proper disconnected communities
- Rumor mechanism assigns per-component misinformation
- Sensing radius is correctly unified at 2 for both agent types
- Exploiters query believed epicenter, explorers query high-uncertainty areas
- AI alignment is only in AIAgent.report_beliefs() (alignment cheating fix verified)
- Relief feedback has correct 15-25 tick delay
- Phase structure (observe/request/decide) executes correctly
- Truthful AI improves belief accuracy (MAE 0.303 vs 0.365 for exploiters)
- Explorers have lower MAE than exploiters (0.196 vs 0.303)

---

## Dead Code to Remove

- `update_belief_bayesian()` — never called
- `smooth_friend_trust()` — never called
- `decay_trust()` — never called
- `Candidate` class — never used
- Imports: `pickle`, `gc`, `csv`, `itertools` — unused
- 50+ commented-out debug print statements
- 7 legacy model files (DisasterBubbles.py, DetectionAction.py, etc.)
- Google Colab mount code in library module (lines 6-13)

---

## Expected vs Observed Behavior Summary

| Expected Pattern | Status | Notes |
|---|---|---|
| High alignment → high exploiter trust in AI | Partial | Q(ai) higher but trust barely changes |
| High alignment → high exploiter AI adoption | Not observed | AI usage ~30% regardless |
| Low alignment → high explorer trust in AI | Not observed | Explorer AI trust ~0.50 regardless |
| Low alignment → high explorer AI adoption | Not observed | Explorer AI usage ~53% regardless |
| Exploiters stick with social networks | Partial | ~64-70% human usage, stable across conditions |
| High alignment amplifies SECI | Weak signal | Correct direction but small magnitude |
| Truthful AI improves accuracy | Observed | MAE 0.303 vs 0.365 for exploiters |
| Explorers have lower MAE | Observed | 0.196 vs 0.303 at alignment=0.1 |

---

## Priority Actions

1. Remove dead `update_belief_bayesian()` to eliminate dual-path confusion
2. Consolidate trust/Q update channels (remove `update_trust_for_accuracy`)
3. Unify reward scaling between `evaluate_information_quality` and `evaluate_pending_info`
4. Reduce explorer trust decay or increase feedback learning rates
5. Add epsilon decay so Q-value-driven choices dominate over time
6. Fix SECI sign convention across all sites
7. Clean up dead code and update documentation
