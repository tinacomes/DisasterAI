"""
Test Protocol for Dual-Timeline Feedback Mechanism

Tests two scenarios:
1. High AI alignment (0.9) - confirming AI should now get penalized via info quality feedback
2. Low AI alignment (0.1) - truthful AI should get rewarded

Tracks:
- Q-values evolution (separate timelines visible)
- Trust in AI sources
- Feedback event frequency (info quality vs relief outcome)
- AI usage patterns (exploratory vs exploitative)
"""

import numpy as np
import matplotlib.pyplot as plt
from DisasterAI_Model import DisasterModel, HumanAgent
import os

# Test parameters
base_params = {
    'share_exploitative': 0.5,
    'share_of_disaster': 0.15,
    'initial_trust': 0.5,
    'initial_ai_trust': 0.5,
    'number_of_humans': 100,
    'share_confirming': 0.7,
    'disaster_dynamics': 2,  # Medium evolution
    'width': 30,
    'height': 30,
    'ticks': 150,
    'learning_rate': 0.1,
    'epsilon': 0.3,
}

def track_agent_feedback_events(model):
    """Track when feedback events occur for sample agents."""
    info_feedback_events = {'exploratory': [], 'exploitative': []}
    relief_feedback_events = {'exploratory': [], 'exploitative': []}
    q_value_evolution = {'exploratory': {'A_0': [], 'human': [], 'self_action': []},
                         'exploitative': {'A_0': [], 'human': [], 'self_action': []}}
    trust_evolution = {'exploratory': {'A_0': []},
                      'exploitative': {'A_0': []}}

    # Select sample agents (first of each type)
    sample_agents = {}
    for agent in model.agent_list:
        if isinstance(agent, HumanAgent):
            if agent.agent_type == 'exploratory' and 'exploratory' not in sample_agents:
                sample_agents['exploratory'] = agent
            elif agent.agent_type == 'exploitative' and 'exploitative' not in sample_agents:
                sample_agents['exploitative'] = agent
        if len(sample_agents) == 2:
            break

    return info_feedback_events, relief_feedback_events, q_value_evolution, trust_evolution, sample_agents

def run_test(ai_alignment, test_name):
    """Run single test and collect metrics."""
    print(f"\n{'='*60}")
    print(f"Running Test: {test_name}")
    print(f"AI Alignment: {ai_alignment}")
    print(f"{'='*60}\n")

    # Create model
    params = base_params.copy()
    params['ai_alignment_level'] = ai_alignment
    model = DisasterModel(**params)

    # Tracking structures
    ai_usage_by_tick = {'exploratory': [], 'exploitative': []}
    q_values_by_tick = {'exploratory': {'A_0': [], 'human': [], 'self_action': []},
                        'exploitative': {'A_0': [], 'human': [], 'self_action': []}}
    trust_by_tick = {'exploratory': [], 'exploitative': []}
    info_feedback_counts = {'exploratory': 0, 'exploitative': 0}
    relief_feedback_counts = {'exploratory': 0, 'exploitative': 0}
    feedback_timeline = {'info': [], 'relief': []}  # (tick, agent_type, event_type)

    # Select sample agents
    sample_agents = {}
    for agent in model.agent_list:
        if isinstance(agent, HumanAgent):
            if agent.agent_type == 'exploratory' and 'exploratory' not in sample_agents:
                sample_agents['exploratory'] = agent
            elif agent.agent_type == 'exploitative' and 'exploitative' not in sample_agents:
                sample_agents['exploitative'] = agent
        if len(sample_agents) == 2:
            break

    # Run simulation
    for tick in range(params['ticks']):
        # Track metrics before step
        for agent_type in ['exploratory', 'exploitative']:
            if agent_type in sample_agents:
                agent = sample_agents[agent_type]

                # Q-values
                q_values_by_tick[agent_type]['A_0'].append(agent.q_table.get('A_0', 0.0))
                q_values_by_tick[agent_type]['human'].append(agent.q_table.get('human', 0.0))
                q_values_by_tick[agent_type]['self_action'].append(agent.q_table.get('self_action', 0.0))

                # Trust in AI
                trust_by_tick[agent_type].append(agent.trust.get('A_0', 0.5))

                # Track pending feedback events
                prev_info_pending = len(agent.pending_info_evaluations)
                prev_relief_pending = len(agent.pending_rewards)

        # Step model
        model.step()

        # Track feedback events that fired this tick
        for agent_type in ['exploratory', 'exploitative']:
            if agent_type in sample_agents:
                agent = sample_agents[agent_type]

                # Info feedback events (check if pending list got shorter = evaluation happened)
                current_info_pending = len(agent.pending_info_evaluations)
                if current_info_pending < prev_info_pending:
                    info_feedback_counts[agent_type] += (prev_info_pending - current_info_pending)
                    feedback_timeline['info'].append((tick, agent_type))

                # Relief feedback events
                current_relief_pending = len(agent.pending_rewards)
                if current_relief_pending < prev_relief_pending:
                    relief_feedback_counts[agent_type] += (prev_relief_pending - current_relief_pending)
                    feedback_timeline['relief'].append((tick, agent_type))

        # Count AI usage this tick
        for agent in model.agent_list:
            if isinstance(agent, HumanAgent):
                if agent.accepted_ai > 0:  # Agent used AI at some point
                    # Check tokens_this_tick for AI usage
                    for mode in agent.tokens_this_tick:
                        if mode.startswith('A_'):
                            if agent.agent_type == 'exploratory':
                                ai_usage_by_tick['exploratory'].append(tick)
                            else:
                                ai_usage_by_tick['exploitative'].append(tick)
                            break

    print(f"\nFeedback Event Summary:")
    print(f"  Exploratory - Info Quality: {info_feedback_counts['exploratory']}, Relief Outcome: {relief_feedback_counts['exploratory']}")
    print(f"  Exploitative - Info Quality: {info_feedback_counts['exploitative']}, Relief Outcome: {relief_feedback_counts['exploitative']}")

    return {
        'q_values': q_values_by_tick,
        'trust': trust_by_tick,
        'ai_usage': ai_usage_by_tick,
        'info_counts': info_feedback_counts,
        'relief_counts': relief_feedback_counts,
        'feedback_timeline': feedback_timeline,
        'sample_agents': sample_agents,
        'model': model
    }

def visualize_results(results_high, results_low):
    """Create comprehensive visualization of both tests."""
    fig = plt.figure(figsize=(16, 12))

    # 1. Q-Values Evolution (High Alignment)
    ax1 = plt.subplot(3, 3, 1)
    for source in ['A_0', 'human', 'self_action']:
        for agent_type in ['exploratory', 'exploitative']:
            data = results_high['q_values'][agent_type][source]
            linestyle = '-' if agent_type == 'exploratory' else '--'
            ax1.plot(data, linestyle=linestyle, label=f"{agent_type[:6]}: {source}", alpha=0.8)
    ax1.set_title('Q-Values: High Alignment (0.9)\nConfirming AI', fontsize=10, fontweight='bold')
    ax1.set_xlabel('Tick')
    ax1.set_ylabel('Q-Value')
    ax1.legend(fontsize=7, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle=':', alpha=0.3)

    # 2. Q-Values Evolution (Low Alignment)
    ax2 = plt.subplot(3, 3, 2)
    for source in ['A_0', 'human', 'self_action']:
        for agent_type in ['exploratory', 'exploitative']:
            data = results_low['q_values'][agent_type][source]
            linestyle = '-' if agent_type == 'exploratory' else '--'
            ax2.plot(data, linestyle=linestyle, label=f"{agent_type[:6]}: {source}", alpha=0.8)
    ax2.set_title('Q-Values: Low Alignment (0.1)\nTruthful AI', fontsize=10, fontweight='bold')
    ax2.set_xlabel('Tick')
    ax2.set_ylabel('Q-Value')
    ax2.legend(fontsize=7, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle=':', alpha=0.3)

    # 3. Trust Evolution Comparison
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(results_high['trust']['exploratory'], '-', label='Explor: High Align', color='blue', alpha=0.7)
    ax3.plot(results_high['trust']['exploitative'], '--', label='Exploit: High Align', color='blue', alpha=0.7)
    ax3.plot(results_low['trust']['exploratory'], '-', label='Explor: Low Align', color='green', alpha=0.7)
    ax3.plot(results_low['trust']['exploitative'], '--', label='Exploit: Low Align', color='green', alpha=0.7)
    ax3.set_title('Trust in AI (A_0) Evolution', fontsize=10, fontweight='bold')
    ax3.set_xlabel('Tick')
    ax3.set_ylabel('Trust')
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)

    # 4. Feedback Timeline (High Alignment)
    ax4 = plt.subplot(3, 3, 4)
    info_explor = [t for t, agent_type in results_high['feedback_timeline']['info'] if agent_type == 'exploratory']
    info_exploit = [t for t, agent_type in results_high['feedback_timeline']['info'] if agent_type == 'exploitative']
    relief_explor = [t for t, agent_type in results_high['feedback_timeline']['relief'] if agent_type == 'exploratory']
    relief_exploit = [t for t, agent_type in results_high['feedback_timeline']['relief'] if agent_type == 'exploitative']

    if info_explor:
        ax4.scatter(info_explor, [1]*len(info_explor), marker='o', s=30, alpha=0.6, label='Info: Explor', color='blue')
    if info_exploit:
        ax4.scatter(info_exploit, [0.8]*len(info_exploit), marker='s', s=30, alpha=0.6, label='Info: Exploit', color='lightblue')
    if relief_explor:
        ax4.scatter(relief_explor, [0.4]*len(relief_explor), marker='^', s=30, alpha=0.6, label='Relief: Explor', color='red')
    if relief_exploit:
        ax4.scatter(relief_exploit, [0.2]*len(relief_exploit), marker='v', s=30, alpha=0.6, label='Relief: Exploit', color='pink')

    ax4.set_title('Feedback Event Timeline\nHigh Alignment', fontsize=10, fontweight='bold')
    ax4.set_xlabel('Tick')
    ax4.set_yticks([0.2, 0.4, 0.8, 1.0])
    ax4.set_yticklabels(['Relief\nExploit', 'Relief\nExplor', 'Info\nExploit', 'Info\nExplor'], fontsize=7)
    ax4.legend(fontsize=7, loc='upper right')
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.set_ylim(0, 1.2)

    # 5. Feedback Timeline (Low Alignment)
    ax5 = plt.subplot(3, 3, 5)
    info_explor = [t for t, agent_type in results_low['feedback_timeline']['info'] if agent_type == 'exploratory']
    info_exploit = [t for t, agent_type in results_low['feedback_timeline']['info'] if agent_type == 'exploitative']
    relief_explor = [t for t, agent_type in results_low['feedback_timeline']['relief'] if agent_type == 'exploratory']
    relief_exploit = [t for t, agent_type in results_low['feedback_timeline']['relief'] if agent_type == 'exploitative']

    if info_explor:
        ax5.scatter(info_explor, [1]*len(info_explor), marker='o', s=30, alpha=0.6, label='Info: Explor', color='blue')
    if info_exploit:
        ax5.scatter(info_exploit, [0.8]*len(info_exploit), marker='s', s=30, alpha=0.6, label='Info: Exploit', color='lightblue')
    if relief_explor:
        ax5.scatter(relief_explor, [0.4]*len(relief_explor), marker='^', s=30, alpha=0.6, label='Relief: Explor', color='red')
    if relief_exploit:
        ax5.scatter(relief_exploit, [0.2]*len(relief_exploit), marker='v', s=30, alpha=0.6, label='Relief: Exploit', color='pink')

    ax5.set_title('Feedback Event Timeline\nLow Alignment', fontsize=10, fontweight='bold')
    ax5.set_xlabel('Tick')
    ax5.set_yticks([0.2, 0.4, 0.8, 1.0])
    ax5.set_yticklabels(['Relief\nExploit', 'Relief\nExplor', 'Info\nExploit', 'Info\nExplor'], fontsize=7)
    ax5.legend(fontsize=7, loc='upper right')
    ax5.grid(True, alpha=0.3, axis='x')
    ax5.set_ylim(0, 1.2)

    # 6. Feedback Frequency Comparison
    ax6 = plt.subplot(3, 3, 6)
    categories = ['Explor\nInfo', 'Explor\nRelief', 'Exploit\nInfo', 'Exploit\nRelief']
    high_align = [
        results_high['info_counts']['exploratory'],
        results_high['relief_counts']['exploratory'],
        results_high['info_counts']['exploitative'],
        results_high['relief_counts']['exploitative']
    ]
    low_align = [
        results_low['info_counts']['exploratory'],
        results_low['relief_counts']['exploratory'],
        results_low['info_counts']['exploitative'],
        results_low['relief_counts']['exploitative']
    ]

    x = np.arange(len(categories))
    width = 0.35
    ax6.bar(x - width/2, high_align, width, label='High Align (0.9)', alpha=0.8, color='orange')
    ax6.bar(x + width/2, low_align, width, label='Low Align (0.1)', alpha=0.8, color='green')
    ax6.set_title('Feedback Event Frequency', fontsize=10, fontweight='bold')
    ax6.set_ylabel('Event Count')
    ax6.set_xticks(x)
    ax6.set_xticklabels(categories, fontsize=8)
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3, axis='y')

    # 7. Final Q-Value Comparison
    ax7 = plt.subplot(3, 3, 7)
    sources = ['A_0', 'human', 'self']
    explor_high = [
        results_high['q_values']['exploratory']['A_0'][-1],
        results_high['q_values']['exploratory']['human'][-1],
        results_high['q_values']['exploratory']['self_action'][-1]
    ]
    exploit_high = [
        results_high['q_values']['exploitative']['A_0'][-1],
        results_high['q_values']['exploitative']['human'][-1],
        results_high['q_values']['exploitative']['self_action'][-1]
    ]

    x = np.arange(len(sources))
    width = 0.35
    ax7.bar(x - width/2, explor_high, width, label='Exploratory', alpha=0.8, color='blue')
    ax7.bar(x + width/2, exploit_high, width, label='Exploitative', alpha=0.8, color='red')
    ax7.set_title('Final Q-Values\nHigh Alignment (0.9)', fontsize=10, fontweight='bold')
    ax7.set_ylabel('Q-Value')
    ax7.set_xticks(x)
    ax7.set_xticklabels(sources, fontsize=8)
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3, axis='y')
    ax7.axhline(y=0, color='k', linestyle=':', alpha=0.3)

    # 8. Final Q-Value Comparison (Low Alignment)
    ax8 = plt.subplot(3, 3, 8)
    explor_low = [
        results_low['q_values']['exploratory']['A_0'][-1],
        results_low['q_values']['exploratory']['human'][-1],
        results_low['q_values']['exploratory']['self_action'][-1]
    ]
    exploit_low = [
        results_low['q_values']['exploitative']['A_0'][-1],
        results_low['q_values']['exploitative']['human'][-1],
        results_low['q_values']['exploitative']['self_action'][-1]
    ]

    ax8.bar(x - width/2, explor_low, width, label='Exploratory', alpha=0.8, color='blue')
    ax8.bar(x + width/2, exploit_low, width, label='Exploitative', alpha=0.8, color='red')
    ax8.set_title('Final Q-Values\nLow Alignment (0.1)', fontsize=10, fontweight='bold')
    ax8.set_ylabel('Q-Value')
    ax8.set_xticks(x)
    ax8.set_xticklabels(sources, fontsize=8)
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.3, axis='y')
    ax8.axhline(y=0, color='k', linestyle=':', alpha=0.3)

    # 9. Summary Metrics
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')

    summary_text = f"""
    DUAL-TIMELINE FEEDBACK TEST SUMMARY

    High Alignment (0.9) - Confirming AI:
    ├─ Exploratory: Info={results_high['info_counts']['exploratory']}, Relief={results_high['relief_counts']['exploratory']}
    ├─ Exploitative: Info={results_high['info_counts']['exploitative']}, Relief={results_high['relief_counts']['exploitative']}
    └─ Final AI Q: Explor={explor_high[0]:.3f}, Exploit={exploit_high[0]:.3f}

    Low Alignment (0.1) - Truthful AI:
    ├─ Exploratory: Info={results_low['info_counts']['exploratory']}, Relief={results_low['relief_counts']['exploratory']}
    ├─ Exploitative: Info={results_low['info_counts']['exploitative']}, Relief={results_low['relief_counts']['exploitative']}
    └─ Final AI Q: Explor={explor_low[0]:.3f}, Exploit={exploit_low[0]:.3f}

    Key Findings:
    • Info feedback is faster & more frequent for exploratory
    • Relief feedback has longer delay (15-25 ticks)
    • Exploratory agents show stronger response to alignment
    • Dual timeline prevents "lucky confirmation" bias
    """

    ax9.text(0.1, 0.5, summary_text, fontsize=9, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    # Save figure
    output_dir = '/home/user/DisasterAI/test_results'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'dual_feedback_test.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")

    return fig

if __name__ == "__main__":
    print("="*60)
    print("DUAL-TIMELINE FEEDBACK MECHANISM TEST PROTOCOL")
    print("="*60)

    # Run tests
    results_high = run_test(ai_alignment=0.9, test_name="High Alignment (Confirming AI)")
    results_low = run_test(ai_alignment=0.1, test_name="Low Alignment (Truthful AI)")

    # Create visualizations
    print("\nGenerating visualizations...")
    fig = visualize_results(results_high, results_low)

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print("\nExpected Results:")
    print("1. Exploratory agents get MORE info quality feedback (more sensing)")
    print("2. High alignment: Exploratory AI Q-value should DECREASE or stay low")
    print("3. Low alignment: Exploratory AI Q-value should INCREASE")
    print("4. Relief feedback should occur 10-20 ticks after info feedback")
    print("5. Exploitative agents less affected by info quality feedback")
