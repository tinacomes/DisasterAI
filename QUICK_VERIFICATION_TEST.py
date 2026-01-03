"""
Quick test to verify info quality feedback fix works
"""

from DisasterAI_Model import DisasterModel, HumanAgent
import random

print("="*70)
print("VERIFYING INFO QUALITY FEEDBACK FIX")
print("="*70)

# Run short test
params = {
    'share_exploitative': 0.5,
    'share_of_disaster': 0.15,
    'initial_trust': 0.3,
    'initial_ai_trust': 0.25,
    'number_of_humans': 20,
    'share_confirming': 0.7,
    'disaster_dynamics': 0,  # Static for testing
    'width': 20,
    'height': 20,
    'ticks': 30,
    'learning_rate': 0.1,
    'epsilon': 0.3,
    'exploit_trust_lr': 0.015,
    'explor_trust_lr': 0.03,
    'ai_alignment_level': 0.9  # Confirming AI
}

model = DisasterModel(**params)

# Track feedback events
exploratory_info = 0
exploitative_info = 0

# Find sample agents
exploratory_agent = None
exploitative_agent = None

for agent in model.agent_list:
    if isinstance(agent, HumanAgent):
        if agent.agent_type == 'exploratory' and not exploratory_agent:
            exploratory_agent = agent
        if agent.agent_type == 'exploitative' and not exploitative_agent:
            exploitative_agent = agent

print(f"\nTracking agents:")
print(f"  Exploratory: {exploratory_agent.unique_id if exploratory_agent else 'None'}")
print(f"  Exploitative: {exploitative_agent.unique_id if exploitative_agent else 'None'}")

# Run simulation
for tick in range(params['ticks']):
    # Track cells_to_verify size
    if tick % 10 == 0 and exploratory_agent:
        print(f"\nTick {tick}:")
        print(f"  Exploratory cells_to_verify: {len(exploratory_agent.cells_to_verify)}")
        print(f"  Exploratory pending_info_evals: {len(exploratory_agent.pending_info_evaluations)}")

    prev_explor_pending = len(exploratory_agent.pending_info_evaluations) if exploratory_agent else 0
    prev_exploit_pending = len(exploitative_agent.pending_info_evaluations) if exploitative_agent else 0

    model.step()

    # Check if feedback fired
    if exploratory_agent:
        curr_pending = len(exploratory_agent.pending_info_evaluations)
        if curr_pending < prev_explor_pending:
            exploratory_info += (prev_explor_pending - curr_pending)

    if exploitative_agent:
        curr_pending = len(exploitative_agent.pending_info_evaluations)
        if curr_pending < prev_exploit_pending:
            exploitative_info += (prev_exploit_pending - curr_pending)

print(f"\n{'='*70}")
print("RESULTS:")
print(f"{'='*70}")
print(f"Exploratory info quality events: {exploratory_info}")
print(f"Exploitative info quality events: {exploitative_info}")

if exploratory_info > 0:
    print(f"\n✓ FIX WORKING: Exploratory agents now get info quality feedback!")
else:
    print(f"\n✗ FIX NOT WORKING: Exploratory agents still get zero feedback")

if exploratory_agent:
    print(f"\nFinal exploratory agent state:")
    print(f"  cells_to_verify remaining: {len(exploratory_agent.cells_to_verify)}")
    print(f"  AI Q-value: {exploratory_agent.q_table.get('A_0', 0.0):.3f}")
