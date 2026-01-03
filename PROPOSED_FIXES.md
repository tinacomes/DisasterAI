# Proposed Fixes for Agent-AI Interaction Model

## Problem Summary

Current reward system uses only **ground truth outcomes** (actual disaster levels), with different weightings:
- Exploratory: 80% actual levels + 20% hit rate
- Exploitative: 20% actual levels + 80% hit rate

**Missing:** No distinction between:
- AI **confirming** agent's belief (validation)
- AI **correcting** agent's belief (information)

Both agent types learn from same signal (actual outcomes), just with different weights.

## Required Changes

### Change 1: Track Confirmation vs Correction

**Where:** In `seek_information()` after receiving AI response

**Add after line 1081 (where last_queried_source_ids is set):**

```python
# Track whether AI confirmed or corrected beliefs
self.last_query_confirmations = {}  # {cell: (ai_report, my_belief, is_confirmation)}

for cell, ai_report in reports.items():
    if cell in self.beliefs and isinstance(self.beliefs[cell], dict):
        my_belief = self.beliefs[cell]['level']
        belief_diff = abs(ai_report - my_belief)

        # Confirmation: AI agrees within ±1 level
        is_confirmation = belief_diff <= 1

        self.last_query_confirmations[cell] = {
            'ai_report': ai_report,
            'my_belief': my_belief,
            'is_confirmation': is_confirmation,
            'source_id': source_id
        }
```

**Also initialize in `__init__` (after line 106):**
```python
self.last_query_confirmations = {}
```

### Change 2: Calculate Confirmation Reward

**Where:** In `process_reward()` when calculating batch_reward

**Replace lines 1284-1297 with:**

```python
# Calculate batch reward based on correctness ratio and actual reward
if cell_rewards:
    avg_actual_reward = sum(cell_rewards) / len(cell_rewards)
    correct_ratio = correct_in_batch / len(cell_rewards) if cell_rewards else 0

    # NEW: Calculate confirmation ratio
    confirmation_count = 0
    correction_count = 0
    confirmation_accurate_count = 0  # Confirmations that were right
    correction_accurate_count = 0     # Corrections that were right

    for cell in cells_to_relieve:
        if cell in self.last_query_confirmations:
            conf_info = self.last_query_confirmations[cell]
            actual_level = self.model.disaster_grid[cell[0], cell[1]]
            ai_was_accurate = abs(conf_info['ai_report'] - actual_level) <= 1

            if conf_info['is_confirmation']:
                confirmation_count += 1
                if ai_was_accurate:
                    confirmation_accurate_count += 1
            else:
                correction_count += 1
                if ai_was_accurate:
                    correction_accurate_count += 1

    total_info = confirmation_count + correction_count

    if total_info > 0:
        confirmation_ratio = confirmation_count / total_info

        # Accuracy of confirmations and corrections
        confirmation_accuracy = (confirmation_accurate_count / confirmation_count
                                 if confirmation_count > 0 else 0.5)
        correction_accuracy = (correction_accurate_count / correction_count
                               if correction_count > 0 else 0.5)
    else:
        confirmation_ratio = 0.5
        confirmation_accuracy = 0.5
        correction_accuracy = 0.5

    # Agent-specific reward calculation
    if self.agent_type == "exploratory":
        # Exploratory agents:
        # - Value accurate corrections highly (new, true information)
        # - Neutral on confirmations (no new info)
        # - Penalize inaccurate corrections

        correction_reward = correction_accuracy * 5.0  # 0 to 5
        confirmation_reward = confirmation_accuracy * 2.0  # 0 to 2 (lower value)
        info_reward = (1 - confirmation_ratio) * correction_reward + confirmation_ratio * confirmation_reward

        # 60% actual outcome, 40% information quality
        batch_reward = 0.6 * avg_actual_reward + 0.4 * info_reward

    else:  # exploitative
        # Exploitative agents:
        # - Value confirmations highly (validation)
        # - Penalize corrections (ego threat)
        # - Care less about accuracy, more about agreement

        confirmation_reward = confirmation_ratio * 5.0  # More confirmations = higher reward

        # Bonus if confirmations turn out accurate (lucky validation)
        if confirmation_count > 0:
            confirmation_reward *= (0.5 + 0.5 * confirmation_accuracy)

        # Penalty for corrections, even if accurate
        correction_penalty = (1 - confirmation_ratio) * 2.0  # Being contradicted

        validation_reward = confirmation_reward - correction_penalty

        # 20% actual outcome, 80% validation
        batch_reward = 0.2 * avg_actual_reward + 0.8 * validation_reward

    # Cap the reward range
    batch_reward = max(-3.0, min(5.0, batch_reward))
else:
    batch_reward = -1.0  # Penalty for targeting nothing
```

### Change 3: Agent-Specific Trust Updates

**Where:** In `process_reward()` trust update section

**Replace lines 1347-1356 with:**

```python
if source_id in self.trust:
    old_trust = self.trust[source_id]

    # Calculate base trust target from relief outcome
    outcome_based_trust = (scaled_reward + 1.0) / 2.0  # Map to [0,1]

    # Adjust based on confirmation/correction for this source
    source_confirmations = 0
    source_corrections = 0
    source_conf_accurate = 0
    source_corr_accurate = 0

    for cell in cells_to_relieve:
        if (cell in self.last_query_confirmations and
            self.last_query_confirmations[cell]['source_id'] == source_id):

            conf_info = self.last_query_confirmations[cell]
            actual_level = self.model.disaster_grid[cell[0], cell[1]]
            ai_accurate = abs(conf_info['ai_report'] - actual_level) <= 1

            if conf_info['is_confirmation']:
                source_confirmations += 1
                if ai_accurate:
                    source_conf_accurate += 1
            else:
                source_corrections += 1
                if ai_accurate:
                    source_corr_accurate += 1

    # Agent-specific trust adjustment
    if self.agent_type == "exploratory":
        # Exploratory: Trust based on ACCURACY of information
        if source_corrections > 0:
            # Corrections were provided - did they help?
            corr_acc_rate = source_corr_accurate / source_corrections
            target_trust = 0.3 + 0.7 * corr_acc_rate  # 0.3 to 1.0
        elif source_confirmations > 0:
            # Only confirmations - moderate trust (no new info)
            conf_acc_rate = source_conf_accurate / source_confirmations
            target_trust = 0.4 + 0.3 * conf_acc_rate  # 0.4 to 0.7
        else:
            # Fallback to outcome-based
            target_trust = outcome_based_trust

        # Bonus for providing corrections (information gain)
        if source_corrections > source_confirmations:
            target_trust = min(1.0, target_trust + 0.1)

    else:  # exploitative
        # Exploitative: Trust based on CONFIRMATION of beliefs
        if source_confirmations > 0:
            # Confirmations provided - very good!
            conf_rate = source_confirmations / (source_confirmations + source_corrections)
            target_trust = 0.5 + 0.5 * conf_rate  # 0.5 to 1.0

            # Bonus if confirmations accurate (lucky validation)
            if source_conf_accurate > 0:
                accuracy_bonus = (source_conf_accurate / source_confirmations) * 0.2
                target_trust = min(1.0, target_trust + accuracy_bonus)
        else:
            # Only corrections - penalty for contradicting
            target_trust = max(0.1, outcome_based_trust - 0.3)

    # Apply trust update
    trust_change = self.trust_learning_rate * (target_trust - old_trust)
    new_trust = max(0.0, min(1.0, old_trust + trust_change))
    self.trust[source_id] = new_trust
```

### Change 4: Clear Confirmation Tracking

**Where:** At end of `process_reward()`, after line 1357

**Add:**
```python
# Clear confirmation tracking after processing
self.last_query_confirmations = {}
```

### Change 5: Info Quality Feedback Enhancement

**Where:** In `evaluate_information_quality()` (lines 397-456)

**Replace lines 420-449 with:**

```python
# Evaluate each pending item
evaluated = []
for (tick_received, source_id, eval_cell, reported_level) in pending:
    if eval_cell in sensed_cells:
        actual_level = sensed_cells[eval_cell]

        # Calculate accuracy reward
        error = abs(reported_level - actual_level)
        if error == 0:
            accuracy_reward = 1.0
        elif error == 1:
            accuracy_reward = 0.5
        elif error == 2:
            accuracy_reward = 0.0
        else:
            accuracy_reward = -0.5 * (error - 2)  # Increasing penalty

        accuracy_reward = max(-1.0, min(1.0, accuracy_reward))

        # Check if this was confirmation or correction
        was_confirmation = False
        if eval_cell in self.beliefs and isinstance(self.beliefs[eval_cell], dict):
            my_belief_at_time = self.beliefs[eval_cell]['level']
            was_confirmation = abs(reported_level - my_belief_at_time) <= 1

        # Agent-specific info quality assessment
        if self.agent_type == "exploratory":
            # Exploratory: Accuracy is key, confirmation is neutral
            if was_confirmation:
                # Confirmation - only reward if accurate
                info_quality_reward = accuracy_reward * 0.5
            else:
                # Correction - reward accuracy highly
                info_quality_reward = accuracy_reward * 1.5
        else:
            # Exploitative: Confirmation is valued, correction penalized
            if was_confirmation:
                # Confirmation - reward even if not perfectly accurate
                info_quality_reward = 0.5 + accuracy_reward * 0.5
            else:
                # Correction - penalize contradiction
                info_quality_reward = accuracy_reward * 0.3 - 0.3

        # Update Q-value for this source
        if source_id in self.q_table:
            old_q = self.q_table[source_id]
            info_learning_rate = self.learning_rate * 0.3
            new_q = old_q + info_learning_rate * (info_quality_reward - old_q)
            self.q_table[source_id] = new_q

            # Also update mode Q-value
            mode = "human" if source_id.startswith("H_") else source_id
            if mode in self.q_table:
                old_mode_q = self.q_table[mode]
                new_mode_q = old_mode_q + info_learning_rate * (info_quality_reward - old_mode_q)
                self.q_table[mode] = new_mode_q

        # Update trust
        if source_id in self.trust:
            old_trust = self.trust[source_id]

            # Map info quality reward to trust adjustment
            trust_adjustment = info_learning_rate * info_quality_reward
            new_trust = max(0.0, min(1.0, old_trust + trust_adjustment))
            self.trust[source_id] = new_trust

        # Mark as evaluated
        evaluated.append((tick_received, source_id, eval_cell, reported_level))
```

## Expected Behavior After Fixes

### With Confirming AI (alignment = 0.9)

**Exploitative agents:**
- Receive confirmations: AI agrees with beliefs
- High confirmation_ratio → high validation_reward
- Trust increases rapidly
- Q-value increases
- **Outcome:** Continue using confirming AI, trust peaks

**Exploratory agents:**
- Receive confirmations: AI agrees with beliefs
- Low correction_accuracy (confirmations don't provide new info)
- info_reward is low (no information gain)
- Trust increases slowly or decreases
- Q-value stagnates or decreases
- **Outcome:** Avoid confirming AI, seek truthful sources

### With Truthful AI (alignment = 0.1)

**Exploitative agents:**
- Receive corrections: AI contradicts beliefs
- Low confirmation_ratio → validation_penalty
- Trust decreases
- Q-value decreases
- **Outcome:** Avoid truthful AI, seek confirming sources

**Exploratory agents:**
- Receive corrections: AI contradicts beliefs
- High correction_accuracy (truthful AI is accurate)
- info_reward is high (information gain)
- Trust increases rapidly
- Q-value increases
- **Outcome:** Continue using truthful AI, trust peaks

## Summary of Changes

1. **Track confirmation vs correction** in `seek_information()`
2. **Calculate confirmation-based rewards** in `process_reward()`:
   - Exploratory: Reward accurate corrections, neutral on confirmations
   - Exploitative: Reward confirmations, penalize corrections
3. **Update trust differently** based on agent type:
   - Exploratory: Trust = accuracy of corrections
   - Exploitative: Trust = rate of confirmations
4. **Enhance info quality feedback** to account for confirmation/correction
5. **Clear tracking** after each reward cycle

This creates **divergent learning** where:
- Exploitative → High alignment AI (confirming)
- Exploratory → Low alignment AI (truthful)

Both achieve high trust, but in **different sources** based on their preferences!
