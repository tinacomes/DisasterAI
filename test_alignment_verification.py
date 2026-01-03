"""
Quick test to verify alignment is working correctly and measure actual belief accuracy.
"""

import numpy as np
from DisasterAI_Model import DisasterModel, HumanAgent, AIAgent

# Create model with high alignment (confirming AI)
print("="*70)
print("ALIGNMENT VERIFICATION TEST")
print("="*70)

params = {
    'share_exploitative': 0.5,
    'share_of_disaster': 0.15,
    'initial_trust': 0.3,
    'initial_ai_trust': 0.25,
    'number_of_humans': 20,  # Smaller for quick test
    'share_confirming': 0.7,
    'disaster_dynamics': 0,  # Static disaster for clearer testing
    'width': 20,
    'height': 20,
    'ticks': 50,
    'learning_rate': 0.1,
    'epsilon': 0.3,
    'exploit_trust_lr': 0.015,
    'explor_trust_lr': 0.03,
    'ai_alignment_level': 0.9  # HIGH ALIGNMENT = CONFIRMING
}

model = DisasterModel(**params)

# Run 1 step to initialize
model.step()

# Find a human agent and an AI agent
human = None
ai = None

for agent in model.agent_list:
    if isinstance(agent, HumanAgent) and human is None:
        human = agent
    if isinstance(agent, AIAgent) and ai is None:
        ai = agent

if not human or not ai:
    print("ERROR: Couldn't find agents!")
    exit(1)

print(f"\nTest Agent: {human.unique_id} ({human.agent_type})")
print(f"AI Agent: {ai.unique_id}")

# Create a test scenario
test_cell = (10, 10)
actual_level = model.disaster_grid[test_cell]

# Give human a WRONG belief about this cell
human.beliefs[test_cell] = {'level': 5, 'confidence': 0.8}

# Make sure AI has sensed the actual value
ai.sensed[test_cell] = actual_level

print(f"\nTest Cell: {test_cell}")
print(f"Actual Level (ground truth): {actual_level}")
print(f"Human Belief: {human.beliefs[test_cell]['level']} (confidence: {human.beliefs[test_cell]['confidence']})")
print(f"AI Sensed (truth): {ai.sensed[test_cell]}")

# Get human's trust in AI
human_trust = human.trust.get(ai.unique_id, 0.25)
print(f"Human Trust in AI: {human_trust:.3f}")

# Query AI with high alignment
print(f"\nQuerying AI with alignment={model.ai_alignment_level}...")
ai_report = ai.report_beliefs(
    interest_point=test_cell,
    query_radius=1,
    caller_beliefs=human.beliefs,
    caller_trust_in_ai=human_trust
)

ai_reported = ai_report.get(test_cell, None)
print(f"AI Reported: {ai_reported}")

# Check if AI confirmed human's wrong belief
if ai_reported is not None:
    print(f"\nVerification:")
    print(f"  Expected (confirming): Close to {human.beliefs[test_cell]['level']} (human's belief)")
    print(f"  Expected (truthful): Close to {actual_level} (ground truth)")
    print(f"  AI Actually Reported: {ai_reported}")

    if abs(ai_reported - human.beliefs[test_cell]['level']) <= 1:
        print(f"  ✓ AI CONFIRMED human belief (alignment working correctly)")
    elif abs(ai_reported - actual_level) <= 1:
        print(f"  ✗ AI REPORTED TRUTH (alignment NOT working!)")
    else:
        print(f"  ? AI reported something else (unexpected)")

    # Calculate what the error would be
    error_from_truth = abs(ai_reported - actual_level)
    print(f"\n  Error from ground truth: {error_from_truth}")
    print(f"  This would give accuracy_reward: ", end="")
    if error_from_truth == 0:
        print(f"+0.03 (current) vs +0.3 (proposed)")
    elif error_from_truth == 1:
        print(f"+0.01 (current) vs +0.1 (proposed)")
    elif error_from_truth == 2:
        print(f"-0.01 (current) vs -0.1 (proposed)")
    else:
        print(f"-0.03 (current) vs -0.3 (proposed)")

# Now test with truthful AI (low alignment)
print(f"\n{'='*70}")
print("TESTING TRUTHFUL AI (alignment=0.1)")
print("="*70)

params['ai_alignment_level'] = 0.1
model2 = DisasterModel(**params)
model2.step()

# Find agents
human2 = None
ai2 = None
for agent in model2.agent_list:
    if isinstance(agent, HumanAgent) and human2 is None:
        human2 = agent
    if isinstance(agent, AIAgent) and ai2 is None:
        ai2 = agent

if human2 and ai2:
    # Same test scenario
    test_cell2 = (10, 10)
    actual_level2 = model2.disaster_grid[test_cell2]
    human2.beliefs[test_cell2] = {'level': 5, 'confidence': 0.8}
    ai2.sensed[test_cell2] = actual_level2

    print(f"Test Cell: {test_cell2}")
    print(f"Actual Level: {actual_level2}")
    print(f"Human Belief: 5 (wrong)")
    print(f"AI Sensed: {actual_level2}")

    human_trust2 = human2.trust.get(ai2.unique_id, 0.25)

    ai_report2 = ai2.report_beliefs(
        interest_point=test_cell2,
        query_radius=1,
        caller_beliefs=human2.beliefs,
        caller_trust_in_ai=human_trust2
    )

    ai_reported2 = ai_report2.get(test_cell2, None)
    print(f"AI Reported: {ai_reported2}")

    if ai_reported2 is not None:
        print(f"\nVerification:")
        print(f"  Expected (truthful): Close to {actual_level2} (ground truth)")
        print(f"  AI Actually Reported: {ai_reported2}")

        if abs(ai_reported2 - actual_level2) <= 1:
            print(f"  ✓ AI REPORTED TRUTH (alignment working correctly)")
        else:
            print(f"  ? Error from truth: {abs(ai_reported2 - actual_level2)}")

print(f"\n{'='*70}")
print("CONCLUSION:")
print("="*70)
print("If alignment is working:")
print("  • High alignment (0.9): AI should confirm human belief (~5), not truth")
print("  • Low alignment (0.1): AI should report truth, ignore human belief")
print("\nIf info quality rewards are too weak:")
print("  • Q-value changes will be ~100x smaller than relief rewards")
print("  • Agents won't learn to avoid confirming AI despite bad information")
