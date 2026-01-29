"""
Test trust dynamics between exploiters and explorers.
Verifies that:
1. Exploiters trust friends more than non-friends
2. Exploiters fall for confirming AI (high alignment)
3. Explorers trust accurate AI (low alignment)
"""

import numpy as np
from DisasterAI_Model import DisasterModel, HumanAgent

def run_test(ai_alignment, ticks=100):
    """Run simulation and track trust evolution."""
    params = {
        'share_exploitative': 0.5,
        'share_of_disaster': 0.15,
        'initial_trust': 0.3,
        'initial_ai_trust': 0.25,
        'number_of_humans': 50,
        'share_confirming': 0.7,
        'disaster_dynamics': 2,
        'width': 20,
        'height': 20,
        'ticks': ticks,
        'learning_rate': 0.1,
        'epsilon': 0.3,
        'ai_alignment_level': ai_alignment,
    }

    model = DisasterModel(**params)

    # Track trust over time
    trust_history = {
        'exploit_ai': [], 'exploit_friend': [], 'exploit_nonfriend': [],
        'explor_ai': [], 'explor_friend': [], 'explor_nonfriend': []
    }

    # Get initial trust values
    exploiters = [a for a in model.agent_list if isinstance(a, HumanAgent) and a.agent_type == "exploitative"]
    explorers = [a for a in model.agent_list if isinstance(a, HumanAgent) and a.agent_type == "exploratory"]

    print(f"\nInitial Trust (alignment={ai_alignment}):")
    print(f"  Exploiter AI trust:      {np.mean([np.mean([a.trust.get(f'A_{i}', 0) for i in range(model.num_ai)]) for a in exploiters]):.3f}")
    print(f"  Exploiter friend trust:  {np.mean([np.mean([a.trust.get(f, 0) for f in a.friends]) for a in exploiters if a.friends]):.3f}")
    print(f"  Explorer AI trust:       {np.mean([np.mean([a.trust.get(f'A_{i}', 0) for i in range(model.num_ai)]) for a in explorers]):.3f}")
    print(f"  Explorer friend trust:   {np.mean([np.mean([a.trust.get(f, 0) for f in a.friends]) for a in explorers if a.friends]):.3f}")

    # Run simulation
    for tick in range(ticks):
        model.step()

        if tick % 20 == 0:
            # Collect trust values
            for agent_type, agents in [('exploit', exploiters), ('explor', explorers)]:
                ai_trusts = []
                friend_trusts = []
                nonfriend_trusts = []

                for a in agents:
                    ai_trusts.append(np.mean([a.trust.get(f'A_{i}', 0) for i in range(model.num_ai)]))
                    if a.friends:
                        friend_trusts.append(np.mean([a.trust.get(f, 0) for f in a.friends if f in a.trust]))
                    nonfriend_trusts.append(np.mean([a.trust.get(h, 0) for h in a.trust if h.startswith('H_') and h not in a.friends]))

                trust_history[f'{agent_type}_ai'].append(np.mean(ai_trusts))
                trust_history[f'{agent_type}_friend'].append(np.mean(friend_trusts) if friend_trusts else 0)
                trust_history[f'{agent_type}_nonfriend'].append(np.mean(nonfriend_trusts))

    print(f"\nFinal Trust (tick {ticks}):")
    print(f"  Exploiter AI trust:        {trust_history['exploit_ai'][-1]:.3f}")
    print(f"  Exploiter friend trust:    {trust_history['exploit_friend'][-1]:.3f}")
    print(f"  Exploiter nonfriend trust: {trust_history['exploit_nonfriend'][-1]:.3f}")
    print(f"  Explorer AI trust:         {trust_history['explor_ai'][-1]:.3f}")
    print(f"  Explorer friend trust:     {trust_history['explor_friend'][-1]:.3f}")
    print(f"  Explorer nonfriend trust:  {trust_history['explor_nonfriend'][-1]:.3f}")

    return trust_history

def main():
    print("="*60)
    print("TRUST DYNAMICS TEST")
    print("="*60)

    print("\n" + "-"*60)
    print("LOW ALIGNMENT (0.1) - Truthful AI")
    print("-"*60)
    low_results = run_test(0.1, ticks=100)

    print("\n" + "-"*60)
    print("HIGH ALIGNMENT (0.9) - Confirming AI")
    print("-"*60)
    high_results = run_test(0.9, ticks=100)

    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)

    print("\nExpected patterns:")
    print("  - Exploiters should trust CONFIRMING AI more (high alignment)")
    print("  - Explorers should trust TRUTHFUL AI more (low alignment)")
    print("  - Exploiters should always trust friends >> non-friends")

    print("\nActual changes (High - Low alignment):")
    print(f"  Exploiter AI trust change:   {high_results['exploit_ai'][-1] - low_results['exploit_ai'][-1]:+.3f}")
    print(f"  Explorer AI trust change:    {high_results['explor_ai'][-1] - low_results['explor_ai'][-1]:+.3f}")

    # Check if patterns match expectations
    exploit_ai_diff = high_results['exploit_ai'][-1] - low_results['exploit_ai'][-1]
    explor_ai_diff = high_results['explor_ai'][-1] - low_results['explor_ai'][-1]

    print("\nPattern verification:")
    if exploit_ai_diff > 0:
        print("  [PASS] Exploiters trust confirming AI MORE")
    else:
        print("  [FAIL] Exploiters should trust confirming AI more")

    if explor_ai_diff < 0:
        print("  [PASS] Explorers trust truthful AI MORE")
    else:
        print("  [FAIL] Explorers should trust truthful AI more")

    # Check friend vs non-friend trust for exploiters
    exploit_friend_gap_low = low_results['exploit_friend'][-1] - low_results['exploit_nonfriend'][-1]
    exploit_friend_gap_high = high_results['exploit_friend'][-1] - high_results['exploit_nonfriend'][-1]

    print(f"\n  Exploiter friend-nonfriend gap (low alignment):  {exploit_friend_gap_low:+.3f}")
    print(f"  Exploiter friend-nonfriend gap (high alignment): {exploit_friend_gap_high:+.3f}")
    if exploit_friend_gap_low > 0.1 and exploit_friend_gap_high > 0.1:
        print("  [PASS] Exploiters maintain strong friend preference")
    else:
        print("  [WARN] Exploiter friend preference may be too weak")

if __name__ == "__main__":
    main()
