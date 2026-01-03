"""
Diagnostic test to understand why exploitative agents prefer AI over humans.
Tracks Q-values and confirmation rates for both source types.
"""

import numpy as np
from DisasterAI_Model import DisasterModel, HumanAgent

def run_diagnostic():
    print("="*70)
    print("Q-VALUE DIAGNOSTIC: Why do exploitative agents prefer AI?")
    print("="*70)

    params = {
        'share_exploitative': 1.0,  # Only exploitative agents
        'share_of_disaster': 0.15,
        'initial_trust': 0.3,
        'initial_ai_trust': 0.25,
        'number_of_humans': 30,
        'share_confirming': 0.5,
        'ai_alignment_level': 0.9,  # Confirming AI
        'disaster_dynamics': 2,
        'width': 20,
        'height': 20,
        'ticks': 50,
        'learning_rate': 0.1,
        'epsilon': 0.3,
        'exploit_trust_lr': 0.015,
        'explor_trust_lr': 0.03,
    }

    model = DisasterModel(**params)

    # Track one sample agent
    sample_agent = None
    for agent in model.agent_list:
        if isinstance(agent, HumanAgent) and agent.agent_type == 'exploitative':
            sample_agent = agent
            break

    q_evolution = {
        'A_0': [],
        'human': [],
        'self_action': []
    }

    confirmation_stats = {
        'AI': {'confirmations': 0, 'corrections': 0},
        'human': {'confirmations': 0, 'corrections': 0}
    }

    for tick in range(params['ticks']):
        # Track Q-values before step
        q_evolution['A_0'].append(sample_agent.q_table.get('A_0', 0.0))
        q_evolution['human'].append(sample_agent.q_table.get('human', 0.0))
        q_evolution['self_action'].append(sample_agent.q_table.get('self_action', 0.0))

        model.step()

        if (tick + 1) % 10 == 0:
            print(f"\nTick {tick + 1}:")
            print(f"  Q(A_0)        = {q_evolution['A_0'][-1]:.4f}")
            print(f"  Q(human)      = {q_evolution['human'][-1]:.4f} (+0.1 bias = {q_evolution['human'][-1] + 0.1:.4f})")
            print(f"  Q(self_action)= {q_evolution['self_action'][-1]:.4f}")

            # Check which source wins
            ai_score = q_evolution['A_0'][-1]
            human_score = q_evolution['human'][-1] + 0.1
            self_score = q_evolution['self_action'][-1] + 0.1

            winner = max([('AI', ai_score), ('human', human_score), ('self', self_score)], key=lambda x: x[1])
            print(f"  → Winner (with biases): {winner[0]} (score={winner[1]:.4f})")

            # Confirmation stats
            print(f"  AI confirmations: {sample_agent.total_confirmations}")
            print(f"  AI corrections: {sample_agent.total_corrections}")

    print("\n" + "="*70)
    print("FINAL ANALYSIS:")
    print("="*70)
    print(f"Final Q-values:")
    print(f"  Q(A_0)   = {q_evolution['A_0'][-1]:.4f}")
    print(f"  Q(human) = {q_evolution['human'][-1]:.4f} (+0.1 bias → {q_evolution['human'][-1] + 0.1:.4f})")
    print(f"  Q(self)  = {q_evolution['self_action'][-1]:.4f} (+0.1 bias → {q_evolution['self_action'][-1] + 0.1:.4f})")

    print(f"\nConfirmation/Correction ratio:")
    total_ai = sample_agent.total_confirmations + sample_agent.total_corrections
    if total_ai > 0:
        conf_rate = sample_agent.total_confirmations / total_ai
        print(f"  AI confirmation rate: {conf_rate:.2%} ({sample_agent.total_confirmations}/{total_ai})")
    else:
        print(f"  No AI queries")

    print(f"\nAgent query history:")
    print(f"  accepted_ai: {sample_agent.accepted_ai}")
    print(f"  accepted_human: {sample_agent.accepted_human}")
    print(f"  accepted_friend: {sample_agent.accepted_friend}")

    # Explanation
    print("\n" + "="*70)
    print("DIAGNOSIS:")
    print("="*70)

    if q_evolution['A_0'][-1] > q_evolution['human'][-1] + 0.1:
        print("✗ PROBLEM: AI Q-value overtook human Q-value despite +0.1 human bias")
        print("\nLikely cause:")
        print("  - Confirming AI (alignment 0.9) validates exploitative agent beliefs")
        print("  - Validation reward = 0.8 × (confirmation_rate × 5.0)")
        print("  - With high confirmation rate, this creates massive Q-value growth")
        print("  - Human bias (+0.1) is too small to compete")
        print("\nSolution:")
        print("  1. Increase human bias for exploitative agents")
        print("  2. Cap AI rewards for exploitative agents")
        print("  3. Make exploitative agents prefer social confirmation over AI")
    else:
        print("✓ Human Q-value winning (as expected)")

    return q_evolution

if __name__ == "__main__":
    q_evolution = run_diagnostic()
