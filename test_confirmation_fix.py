"""
Quick test to verify confirmation/correction tracking works correctly.

Tests whether exploitative and exploratory agents now show divergent trust patterns
based on AI alignment (confirming vs truthful).
"""

import numpy as np
import matplotlib.pyplot as plt
from DisasterAI_Model import DisasterModel, HumanAgent

def run_quick_test(ai_alignment, test_name):
    """Run quick test and track trust evolution by agent type."""
    print(f"\n{'='*60}")
    print(f"Test: {test_name}")
    print(f"AI Alignment: {ai_alignment}")
    print(f"{'='*60}\n")

    params = {
        'share_exploitative': 0.5,
        'share_of_disaster': 0.15,
        'initial_trust': 0.3,
        'initial_ai_trust': 0.25,
        'number_of_humans': 50,
        'share_confirming': 0.5,
        'ai_alignment_level': ai_alignment,
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

    # Separate agents by type
    agents_by_type = {'exploratory': [], 'exploitative': []}
    for agent in model.agent_list:
        if isinstance(agent, HumanAgent):
            agents_by_type[agent.agent_type].append(agent)

    # Track trust in primary AI (A_0) over time
    trust_evolution = {
        'exploratory': [],
        'exploitative': []
    }

    for tick in range(params['ticks']):
        model.step()

        # Track trust
        for agent_type in ['exploratory', 'exploitative']:
            agents = agents_by_type[agent_type]
            if len(agents) > 0:
                # Average trust in A_0 across all agents of this type
                avg_trust = np.mean([agent.trust.get('A_0', 0.25) for agent in agents])
                trust_evolution[agent_type].append(avg_trust)

        if (tick + 1) % 25 == 0:
            print(f"Tick {tick + 1}:")
            print(f"  Exploratory avg trust in A_0: {trust_evolution['exploratory'][-1]:.3f}")
            print(f"  Exploitative avg trust in A_0: {trust_evolution['exploitative'][-1]:.3f}")

    # Get final confirmation/correction counts
    confirmation_counts = {
        'exploratory': {'confirmations': 0, 'corrections': 0},
        'exploitative': {'confirmations': 0, 'corrections': 0}
    }

    for agent_type in ['exploratory', 'exploitative']:
        for agent in agents_by_type[agent_type]:
            confirmation_counts[agent_type]['confirmations'] += agent.total_confirmations
            confirmation_counts[agent_type]['corrections'] += agent.total_corrections

    # Print final results
    print(f"\n{'='*60}")
    print("FINAL RESULTS:")
    print(f"{'='*60}")
    print(f"Exploratory agents:")
    print(f"  Final trust in A_0: {trust_evolution['exploratory'][-1]:.3f}")
    print(f"  Initial trust: {trust_evolution['exploratory'][0]:.3f}")
    print(f"  Change: {trust_evolution['exploratory'][-1] - trust_evolution['exploratory'][0]:+.3f}")
    print(f"  Confirmations received: {confirmation_counts['exploratory']['confirmations']}")
    print(f"  Corrections received: {confirmation_counts['exploratory']['corrections']}")

    print(f"\nExploitative agents:")
    print(f"  Final trust in A_0: {trust_evolution['exploitative'][-1]:.3f}")
    print(f"  Initial trust: {trust_evolution['exploitative'][0]:.3f}")
    print(f"  Change: {trust_evolution['exploitative'][-1] - trust_evolution['exploitative'][0]:+.3f}")
    print(f"  Confirmations received: {confirmation_counts['exploitative']['confirmations']}")
    print(f"  Corrections received: {confirmation_counts['exploitative']['corrections']}")

    return trust_evolution, confirmation_counts


def main():
    print("="*60)
    print("CONFIRMATION/CORRECTION TRACKING TEST")
    print("="*60)
    print("\nTesting if agents now differentiate between:")
    print("  - Confirming AI (alignment 0.9) - validates beliefs")
    print("  - Truthful AI (alignment 0.1) - reports ground truth")
    print("\nExpected behavior:")
    print("  Confirming AI: Exploitative trust ↑, Exploratory trust →/↓")
    print("  Truthful AI:   Exploitative trust ↓, Exploratory trust ↑")

    # Test 1: Confirming AI
    trust_confirm, conf_confirm = run_quick_test(0.9, "Confirming AI")

    # Test 2: Truthful AI
    trust_truthful, conf_truthful = run_quick_test(0.1, "Truthful AI")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Confirming AI
    ax1 = axes[0]
    ax1.plot(trust_confirm['exploratory'], 'b-', linewidth=2, label='Exploratory', alpha=0.8)
    ax1.plot(trust_confirm['exploitative'], 'r-', linewidth=2, label='Exploitative', alpha=0.8)
    ax1.axhline(0.25, color='k', linestyle='--', alpha=0.3, label='Initial')
    ax1.set_title('Confirming AI (0.9)\nExpect: Exploit ↑, Explor →/↓', fontweight='bold')
    ax1.set_xlabel('Tick')
    ax1.set_ylabel('Average Trust in A_0')
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Truthful AI
    ax2 = axes[1]
    ax2.plot(trust_truthful['exploratory'], 'b-', linewidth=2, label='Exploratory', alpha=0.8)
    ax2.plot(trust_truthful['exploitative'], 'r-', linewidth=2, label='Exploitative', alpha=0.8)
    ax2.axhline(0.25, color='k', linestyle='--', alpha=0.3, label='Initial')
    ax2.set_title('Truthful AI (0.1)\nExpect: Exploit ↓, Explor ↑', fontweight='bold')
    ax2.set_xlabel('Tick')
    ax2.set_ylabel('Average Trust in A_0')
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('test_results/confirmation_fix_test.png', dpi=150, bbox_inches='tight')
    print("\n" + "="*60)
    print("Test complete! Visualization saved to:")
    print("  test_results/confirmation_fix_test.png")
    print("="*60)

    # Check if divergence occurred
    print("\n" + "="*60)
    print("DIVERGENCE CHECK:")
    print("="*60)

    # Confirming AI
    confirm_explor_change = trust_confirm['exploratory'][-1] - trust_confirm['exploratory'][0]
    confirm_exploit_change = trust_confirm['exploitative'][-1] - trust_confirm['exploitative'][0]

    print(f"\nConfirming AI (0.9):")
    print(f"  Exploratory change: {confirm_explor_change:+.3f}")
    print(f"  Exploitative change: {confirm_exploit_change:+.3f}")
    if confirm_exploit_change > confirm_explor_change + 0.05:
        print("  ✅ CORRECT: Exploitative trust AI more than exploratory")
    else:
        print("  ❌ UNEXPECTED: Both agent types behaving similarly")

    # Truthful AI
    truthful_explor_change = trust_truthful['exploratory'][-1] - trust_truthful['exploratory'][0]
    truthful_exploit_change = trust_truthful['exploitative'][-1] - trust_truthful['exploitative'][0]

    print(f"\nTruthful AI (0.1):")
    print(f"  Exploratory change: {truthful_explor_change:+.3f}")
    print(f"  Exploitative change: {truthful_exploit_change:+.3f}")
    if truthful_explor_change > truthful_exploit_change + 0.05:
        print("  ✅ CORRECT: Exploratory trust AI more than exploitative")
    else:
        print("  ❌ UNEXPECTED: Both agent types behaving similarly")


if __name__ == "__main__":
    import os
    os.makedirs('test_results', exist_ok=True)
    main()
