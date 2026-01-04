"""Debug script to track what modes explorers actually choose."""
import sys
from DisasterAI_Model import DisasterModel, HumanAgent

# Simple test parameters (matching test_dual_feedback.py)
params = {
    'share_exploitative': 0.5,
    'share_of_disaster': 0.15,
    'initial_trust': 0.3,
    'initial_ai_trust': 0.25,
    'number_of_humans': 20,  # Smaller for debug
    'share_confirming': 0.7,
    'disaster_dynamics': 2,
    'width': 30,
    'height': 30,
    'ticks': 50,
    'learning_rate': 0.1,
    'epsilon': 0.3,
    'ai_alignment_level': 0.9,  # High alignment (confirming AI)
    'exploit_trust_lr': 0.015,
    'explor_trust_lr': 0.03
}

model = DisasterModel(**params)

# Find first exploratory agent
explorer = None
for agent in model.agent_list:
    if isinstance(agent, HumanAgent) and agent.agent_type == 'exploratory':
        explorer = agent
        break

if not explorer:
    print("No exploratory agent found!")
    sys.exit(1)

print(f"Tracking agent {explorer.unique_id} (exploratory)")
print("\nMode choices over 50 ticks:")
print("Tick | Mode | Q-values (self/human/ai) | Pending Info | Significant Changes")
print("-" * 85)

mode_counts = {'self_action': 0, 'human': 0, 'ai': 0}
significant_changes = 0

for tick in range(50):
    # Track before step
    q_self = explorer.q_table.get('self_action', 0)
    q_human = explorer.q_table.get('human', 0)
    q_ai = explorer.q_table.get('ai', 0)
    pending_before = len(explorer.pending_info_evaluations)

    # Step
    model.step()

    # Check what mode was chosen (from tokens_this_tick)
    mode = None
    if hasattr(explorer, 'tokens_this_tick'):
        for m in explorer.tokens_this_tick:
            if explorer.tokens_this_tick[m] > 0:
                mode = m
                mode_counts[mode] += 1
                break

    pending_after = len(explorer.pending_info_evaluations)
    change = pending_after - pending_before
    if change > 0:
        significant_changes += 1

    if tick % 5 == 0 or change > 0:  # Print every 5 ticks or when info is added
        print(f"{tick:4d} | {mode:11s} | {q_self:5.2f}/{q_human:5.2f}/{q_ai:5.2f} | "
              f"{pending_after:12d} | {'+' if change > 0 else ''}{change}")

print("\n" + "=" * 85)
print(f"\nMode distribution:")
print(f"  self_action: {mode_counts['self_action']/50*100:.1f}% ({mode_counts['self_action']}/50)")
print(f"  human:       {mode_counts['human']/50*100:.1f}% ({mode_counts['human']}/50)")
print(f"  ai:          {mode_counts['ai']/50*100:.1f}% ({mode_counts['ai']}/50)")
print(f"\nSignificant belief changes (info added to pending): {significant_changes}")
print(f"Pending evaluations at end: {len(explorer.pending_info_evaluations)}")

# Check AI-reliant status
print(f"\n--- AI-Reliant Status Check ---")
print(f"accum_calls_total: {explorer.accum_calls_total}")
print(f"accum_calls_ai: {explorer.accum_calls_ai}")
print(f"accum_calls_human: {explorer.accum_calls_human}")
if explorer.accum_calls_total > 0:
    ai_ratio = explorer.accum_calls_ai / explorer.accum_calls_total
    print(f"AI ratio: {ai_ratio:.2%} ({explorer.accum_calls_ai}/{explorer.accum_calls_total})")
    min_threshold = 0.25
    min_calls = 10
    print(f"Threshold: {min_threshold*100:.0f}% of at least {min_calls} calls")
    print(f"Qualifies as AI-reliant: {explorer.accum_calls_total >= min_calls and ai_ratio >= min_threshold}")
else:
    print("No queries made!")
