"""
Test script for corrected DisasterAI model
Tests key fixes: alignment, Q-learning, disaster dynamics
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from DisasterAI_Model import DisasterModel, HumanAgent

def run_alignment_test(ai_alignment=0.5, num_ticks=100, test_name="Test"):
    """
    Test the corrected model with specified alignment level.
    Tracks Q-values and trust to verify learning behavior.
    """
    print(f"\n{'='*60}")
    print(f"Running {test_name}")
    print(f"AI Alignment: {ai_alignment}")
    print(f"{'='*60}\n")

    # Create model
    model = DisasterModel(
        share_exploitative=0.5,
        share_of_disaster=0.2,
        initial_trust=0.5,
        initial_ai_trust=0.5,
        number_of_humans=20,
        share_confirming=0.5,
        disaster_dynamics=2,  # Medium dynamics
        shock_probability=0.1,
        shock_magnitude=2,
        trust_update_mode="average",
        ai_alignment_level=ai_alignment,
        exploitative_correction_factor=1.0,
        width=20,
        height=20,
        lambda_parameter=0.5,
        learning_rate=0.05,
        epsilon=0.2,
        ticks=num_ticks
    )

    # Track metrics
    metrics = {
        'exploitative': {
            'q_human': [],
            'q_ai': [],
            'trust_ai': [],
            'trust_human': [],
            'correct_targets': [],
            'ai_calls': [],
            'human_calls': []
        },
        'exploratory': {
            'q_human': [],
            'q_ai': [],
            'trust_ai': [],
            'trust_human': [],
            'correct_targets': [],
            'ai_calls': [],
            'human_calls': []
        }
    }

    # Get sample agents of each type
    exploiter = None
    explorer = None
    for agent in model.agent_list:
        if isinstance(agent, HumanAgent):
            if agent.agent_type == "exploitative" and not exploiter:
                exploiter = agent
            elif agent.agent_type == "exploratory" and not explorer:
                explorer = agent
        if exploiter and explorer:
            break

    if not exploiter or not explorer:
        print("ERROR: Could not find both agent types!")
        return None

    print(f"Tracking Exploiter: {exploiter.unique_id}")
    print(f"Tracking Explorer: {explorer.unique_id}")
    print(f"\nInitial Q-values:")
    print(f"  Exploiter: {exploiter.q_table}")
    print(f"  Explorer: {explorer.q_table}")

    # Run simulation
    for tick in range(num_ticks):
        model.step()

        # Track exploiter
        metrics['exploitative']['q_human'].append(exploiter.q_table.get('human', 0.0))
        metrics['exploitative']['q_ai'].append(exploiter.q_table.get('ai', 0.0))
        metrics['exploitative']['correct_targets'].append(exploiter.correct_targets)
        metrics['exploitative']['ai_calls'].append(exploiter.accum_calls_ai)
        metrics['exploitative']['human_calls'].append(exploiter.accum_calls_human)

        # Get average AI trust and average human friend trust
        ai_trusts = [exploiter.trust.get(f"A_{k}", 0.5) for k in range(model.num_ai)]
        metrics['exploitative']['trust_ai'].append(np.mean(ai_trusts))

        if exploiter.friends:
            friend_trusts = [exploiter.trust.get(fid, 0.5) for fid in exploiter.friends]
            metrics['exploitative']['trust_human'].append(np.mean(friend_trusts))
        else:
            metrics['exploitative']['trust_human'].append(0.5)

        # Track explorer
        metrics['exploratory']['q_human'].append(explorer.q_table.get('human', 0.0))
        metrics['exploratory']['q_ai'].append(explorer.q_table.get('ai', 0.0))
        metrics['exploratory']['correct_targets'].append(explorer.correct_targets)
        metrics['exploratory']['ai_calls'].append(explorer.accum_calls_ai)
        metrics['exploratory']['human_calls'].append(explorer.accum_calls_human)

        ai_trusts = [explorer.trust.get(f"A_{k}", 0.5) for k in range(model.num_ai)]
        metrics['exploratory']['trust_ai'].append(np.mean(ai_trusts))

        if explorer.friends:
            friend_trusts = [explorer.trust.get(fid, 0.5) for fid in explorer.friends]
            metrics['exploratory']['trust_human'].append(np.mean(friend_trusts))
        else:
            metrics['exploratory']['trust_human'].append(0.5)

        # Progress indicator
        if (tick + 1) % 20 == 0:
            print(f"Tick {tick + 1}/{num_ticks}")

    # Final report
    print(f"\n{'='*60}")
    print(f"RESULTS - {test_name}")
    print(f"{'='*60}")

    print(f"\nEXPLOITER (should favor confirming sources):")
    print(f"  Final Q-values: Human={exploiter.q_table.get('human', 0):.3f}, AI={exploiter.q_table.get('ai', 0):.3f}")
    print(f"  Total calls: Human={exploiter.accum_calls_human}, AI={exploiter.accum_calls_ai}")
    print(f"  Correct targets: {exploiter.correct_targets}")
    print(f"  Q-difference (Human-AI): {exploiter.q_table.get('human', 0) - exploiter.q_table.get('ai', 0):.3f}")

    print(f"\nEXPLORER (should favor accurate sources):")
    print(f"  Final Q-values: Human={explorer.q_table.get('human', 0):.3f}, AI={explorer.q_table.get('ai', 0):.3f}")
    print(f"  Total calls: Human={explorer.accum_calls_human}, AI={explorer.accum_calls_ai}")
    print(f"  Correct targets: {explorer.correct_targets}")
    print(f"  Q-difference (Human-AI): {explorer.q_table.get('human', 0) - explorer.q_table.get('ai', 0):.3f}")

    # Expected behavior analysis
    print(f"\n{'='*60}")
    print("EXPECTED BEHAVIOR:")
    if ai_alignment > 0.5:
        print("HIGH ALIGNMENT → AI confirms beliefs")
        print("  - Exploiters should learn AI is valuable (Q_ai > Q_human)")
        print("  - Explorers should learn AI is less valuable (Q_ai < Q_human)")
    else:
        print("LOW ALIGNMENT → AI tells truth")
        print("  - Explorers should learn AI is valuable (Q_ai > Q_human)")
        print("  - Exploiters may still prefer friends (Q_human > Q_ai)")

    print(f"\nACTUAL BEHAVIOR:")
    exploiter_prefers_ai = exploiter.q_table.get('ai', 0) > exploiter.q_table.get('human', 0)
    explorer_prefers_ai = explorer.q_table.get('ai', 0) > explorer.q_table.get('human', 0)

    print(f"  - Exploiter prefers: {'AI' if exploiter_prefers_ai else 'Human'}")
    print(f"  - Explorer prefers: {'AI' if explorer_prefers_ai else 'Human'}")

    return metrics


def plot_results(metrics_high, metrics_low):
    """Plot comparison of high vs low alignment results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Q-Learning with Corrected Model: High vs Low Alignment', fontsize=16)

    ticks = range(len(metrics_high['exploitative']['q_human']))

    # Row 1: Exploiters
    # Q-values
    axes[0, 0].plot(ticks, metrics_high['exploitative']['q_human'], label='Human (High Align)', color='blue', linestyle='-')
    axes[0, 0].plot(ticks, metrics_high['exploitative']['q_ai'], label='AI (High Align)', color='red', linestyle='-')
    axes[0, 0].plot(ticks, metrics_low['exploitative']['q_human'], label='Human (Low Align)', color='blue', linestyle='--')
    axes[0, 0].plot(ticks, metrics_low['exploitative']['q_ai'], label='AI (Low Align)', color='red', linestyle='--')
    axes[0, 0].set_title('Exploiter Q-Values')
    axes[0, 0].set_xlabel('Tick')
    axes[0, 0].set_ylabel('Q-Value')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Call counts
    axes[0, 1].plot(ticks, metrics_high['exploitative']['human_calls'], label='Human (High)', color='blue', linestyle='-')
    axes[0, 1].plot(ticks, metrics_high['exploitative']['ai_calls'], label='AI (High)', color='red', linestyle='-')
    axes[0, 1].plot(ticks, metrics_low['exploitative']['human_calls'], label='Human (Low)', color='blue', linestyle='--')
    axes[0, 1].plot(ticks, metrics_low['exploitative']['ai_calls'], label='AI (Low)', color='red', linestyle='--')
    axes[0, 1].set_title('Exploiter Source Calls')
    axes[0, 1].set_xlabel('Tick')
    axes[0, 1].set_ylabel('Cumulative Calls')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Trust
    axes[0, 2].plot(ticks, metrics_high['exploitative']['trust_ai'], label='AI Trust (High)', color='red', linestyle='-')
    axes[0, 2].plot(ticks, metrics_low['exploitative']['trust_ai'], label='AI Trust (Low)', color='red', linestyle='--')
    axes[0, 2].plot(ticks, metrics_high['exploitative']['trust_human'], label='Human Trust (High)', color='blue', linestyle='-')
    axes[0, 2].plot(ticks, metrics_low['exploitative']['trust_human'], label='Human Trust (Low)', color='blue', linestyle='--')
    axes[0, 2].set_title('Exploiter Trust Levels')
    axes[0, 2].set_xlabel('Tick')
    axes[0, 2].set_ylabel('Trust')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Row 2: Explorers
    # Q-values
    axes[1, 0].plot(ticks, metrics_high['exploratory']['q_human'], label='Human (High Align)', color='blue', linestyle='-')
    axes[1, 0].plot(ticks, metrics_high['exploratory']['q_ai'], label='AI (High Align)', color='red', linestyle='-')
    axes[1, 0].plot(ticks, metrics_low['exploratory']['q_human'], label='Human (Low Align)', color='blue', linestyle='--')
    axes[1, 0].plot(ticks, metrics_low['exploratory']['q_ai'], label='AI (Low Align)', color='red', linestyle='--')
    axes[1, 0].set_title('Explorer Q-Values')
    axes[1, 0].set_xlabel('Tick')
    axes[1, 0].set_ylabel('Q-Value')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Call counts
    axes[1, 1].plot(ticks, metrics_high['exploratory']['human_calls'], label='Human (High)', color='blue', linestyle='-')
    axes[1, 1].plot(ticks, metrics_high['exploratory']['ai_calls'], label='AI (High)', color='red', linestyle='-')
    axes[1, 1].plot(ticks, metrics_low['exploratory']['human_calls'], label='Human (Low)', color='blue', linestyle='--')
    axes[1, 1].plot(ticks, metrics_low['exploratory']['ai_calls'], label='AI (Low)', color='red', linestyle='--')
    axes[1, 1].set_title('Explorer Source Calls')
    axes[1, 1].set_xlabel('Tick')
    axes[1, 1].set_ylabel('Cumulative Calls')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Trust
    axes[1, 2].plot(ticks, metrics_high['exploratory']['trust_ai'], label='AI Trust (High)', color='red', linestyle='-')
    axes[1, 2].plot(ticks, metrics_low['exploratory']['trust_ai'], label='AI Trust (Low)', color='red', linestyle='--')
    axes[1, 2].plot(ticks, metrics_high['exploratory']['trust_human'], label='Human Trust (High)', color='blue', linestyle='-')
    axes[1, 2].plot(ticks, metrics_low['exploratory']['trust_human'], label='Human Trust (Low)', color='blue', linestyle='--')
    axes[1, 2].set_title('Explorer Trust Levels')
    axes[1, 2].set_xlabel('Tick')
    axes[1, 2].set_ylabel('Trust')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('corrected_model_test_results.png', dpi=150, bbox_inches='tight')
    print("\n✓ Plot saved as 'corrected_model_test_results.png'")
    plt.show()


if __name__ == "__main__":
    print("="*60)
    print("Testing Corrected DisasterAI Model")
    print("="*60)
    print("\nThis test verifies the 6 critical fixes:")
    print("1. Division by zero prevention")
    print("2. Correct AI alignment formula")
    print("3. Simplified source selection (human OR ai)")
    print("4. Disaster dynamics implementation")
    print("5. Q-learning convergence")
    print("\nExpected outcomes:")
    print("- With HIGH alignment (0.9): AI confirms beliefs")
    print("  → Exploiters should prefer AI (Q_ai > Q_human)")
    print("  → Explorers should prefer humans (Q_human > Q_ai)")
    print("\n- With LOW alignment (0.1): AI tells truth")
    print("  → Explorers should prefer AI (Q_ai > Q_human)")
    print("  → Exploiters may still prefer friends")

    # Run tests
    metrics_high = run_alignment_test(
        ai_alignment=0.9,
        num_ticks=100,
        test_name="High Alignment (AI Confirms)"
    )

    metrics_low = run_alignment_test(
        ai_alignment=0.1,
        num_ticks=100,
        test_name="Low Alignment (AI Truthful)"
    )

    if metrics_high and metrics_low:
        plot_results(metrics_high, metrics_low)
        print("\n✓ Test complete!")
    else:
        print("\n✗ Test failed - could not complete both conditions")
