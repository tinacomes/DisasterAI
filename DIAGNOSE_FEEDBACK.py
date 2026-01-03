"""
Diagnose why exploratory agents get 0 info quality feedback in large test
"""

from DisasterAI_Model import DisasterModel, HumanAgent
import random

print("="*70)
print("DIAGNOSTIC: Comparing small vs large test")
print("="*70)

def run_diagnostic(num_agents, grid_size, disaster_dynamics, ticks, test_name):
    print(f"\n{'='*70}")
    print(f"TEST: {test_name}")
    print(f"  Agents: {num_agents}, Grid: {grid_size}x{grid_size}, Dynamics: {disaster_dynamics}")
    print(f"{'='*70}")

    params = {
        'share_exploitative': 0.5,
        'share_of_disaster': 0.15,
        'initial_trust': 0.3,
        'initial_ai_trust': 0.25,
        'number_of_humans': num_agents,
        'share_confirming': 0.7,
        'disaster_dynamics': disaster_dynamics,
        'width': grid_size,
        'height': grid_size,
        'ticks': ticks,
        'learning_rate': 0.1,
        'epsilon': 0.3,
        'exploit_trust_lr': 0.015,
        'explor_trust_lr': 0.03,
        'ai_alignment_level': 0.9  # Confirming AI
    }

    model = DisasterModel(**params)

    # Find exploratory agent
    exploratory_agent = None
    for agent in model.agent_list:
        if isinstance(agent, HumanAgent) and agent.agent_type == 'exploratory':
            exploratory_agent = agent
            break

    # Track metrics
    ai_queries = 0
    cells_added = 0
    info_events = 0

    for tick in range(ticks):
        prev_cells_to_verify = len(exploratory_agent.cells_to_verify)
        prev_pending = len(exploratory_agent.pending_info_evaluations)

        # Check if agent queries AI this tick
        old_q_count = ai_queries

        model.step()

        # Check if cells were added
        new_cells_to_verify = len(exploratory_agent.cells_to_verify)
        if new_cells_to_verify > prev_cells_to_verify:
            cells_added += (new_cells_to_verify - prev_cells_to_verify)

        # Check if info events fired
        new_pending = len(exploratory_agent.pending_info_evaluations)
        if new_pending < prev_pending:
            info_events += (prev_pending - new_pending)

        # Sample reporting
        if tick % 10 == 0:
            ai_trust = exploratory_agent.q_table.get('A_0', 0.0)
            print(f"  Tick {tick:3d}: cells_to_verify={new_cells_to_verify:3d}, "
                  f"pending={new_pending:3d}, info_events={info_events:3d}, "
                  f"AI_Q={ai_trust:.3f}")

    print(f"\nFINAL RESULTS:")
    print(f"  Total cells added to verify: {cells_added}")
    print(f"  Total info quality events: {info_events}")
    print(f"  Final cells_to_verify: {len(exploratory_agent.cells_to_verify)}")
    print(f"  Final AI Q-value: {exploratory_agent.q_table.get('A_0', 0.0):.3f}")

    return info_events

# Run small test (like quick verification)
small_events = run_diagnostic(
    num_agents=20,
    grid_size=20,
    disaster_dynamics=0,
    ticks=30,
    test_name="SMALL (Quick Test Parameters)"
)

# Run large test (like full test)
large_events = run_diagnostic(
    num_agents=100,
    grid_size=30,
    disaster_dynamics=2,
    ticks=50,  # Shorter for faster diagnosis
    test_name="LARGE (Full Test Parameters)"
)

print(f"\n{'='*70}")
print("COMPARISON:")
print(f"{'='*70}")
print(f"Small test info events: {small_events}")
print(f"Large test info events: {large_events}")

if small_events > 0 and large_events == 0:
    print("\n⚠️  CONFIRMED: Fix works in small test but fails in large test")
    print("    Investigating scaling issue...")
elif large_events > 0:
    print("\n✓ Fix works in both tests!")
else:
    print("\n✗ Fix not working in either test")
