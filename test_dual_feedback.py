"""
Test Protocol for Dual-Timeline Feedback Mechanism

Tests two scenarios:
1. High AI alignment (0.9) - confirming AI should now get penalized via info quality feedback
2. Low AI alignment (0.1) - truthful AI should get rewarded

Tracks:
- Q-values evolution (separate timelines visible)
- Trust in AI sources
- Feedback event frequency (info quality vs relief outcome)
- SECI (Social Echo Chamber Index) evolution
- AECI (AI Echo Chamber Index) evolution
- Belief accuracy (MAE) evolution
- AI vs friend preference over time
"""

import numpy as np
import matplotlib.pyplot as plt
from DisasterAI_Model import DisasterModel, HumanAgent
import os

# Test parameters
base_params = {
    'share_exploitative': 0.5,
    'share_of_disaster': 0.15,
    'initial_trust': 0.3,
    'initial_ai_trust': 0.25,
    'number_of_humans': 100,
    'share_confirming': 0.7,
    'disaster_dynamics': 2,
    'width': 30,
    'height': 30,
    'ticks': 150,
    'learning_rate': 0.1,
    'epsilon': 0.3,
    'exploit_trust_lr': 0.015,
    'explor_trust_lr': 0.03,
}

def run_test(ai_alignment, test_name):
    """Run single test and collect all metrics including SECI, AECI, MAE, AI preference."""
    print(f"\n{'='*60}")
    print(f"Running Test: {test_name}")
    print(f"AI Alignment: {ai_alignment}")
    print(f"{'='*60}\n")

    params = base_params.copy()
    params['ai_alignment_level'] = ai_alignment
    model = DisasterModel(**params)

    # --- Tracking structures ---
    # Q-values and trust (sample agents)
    q_values_by_tick = {'exploratory': {'ai': [], 'human': [], 'self_action': []},
                        'exploitative': {'ai': [], 'human': [], 'self_action': []}}
    trust_by_tick = {'exploratory': [], 'exploitative': []}

    # Feedback events (sample agents)
    info_feedback_counts = {'exploratory': 0, 'exploitative': 0}
    relief_feedback_counts = {'exploratory': 0, 'exploitative': 0}
    feedback_timeline = {'info': [], 'relief': []}

    # Echo chamber indices (from model, every tick)
    seci_by_tick = {'exploit': [], 'explor': []}
    aeci_by_tick = {'exploit': [], 'explor': []}

    # Belief accuracy MAE (all agents, every 10 ticks)
    belief_accuracy = {'exploratory': [], 'exploitative': []}

    # AI vs friend preference (all agents, every 10 ticks)
    ai_preference = {'exploratory': [], 'exploitative': []}

    # Select sample agents for Q-value / feedback tracking
    sample_agents = {}
    for agent in model.agent_list:
        if isinstance(agent, HumanAgent):
            if agent.agent_type == 'exploratory' and 'exploratory' not in sample_agents:
                sample_agents['exploratory'] = agent
            elif agent.agent_type == 'exploitative' and 'exploitative' not in sample_agents:
                sample_agents['exploitative'] = agent
        if len(sample_agents) == 2:
            break

    # --- Run simulation ---
    for tick in range(params['ticks']):
        # Track Q-values and trust before step
        prev_info_pending = {}
        prev_relief_pending = {}
        for agent_type in ['exploratory', 'exploitative']:
            if agent_type in sample_agents:
                agent = sample_agents[agent_type]
                q_values_by_tick[agent_type]['ai'].append(agent.q_table.get('ai', 0.0))
                q_values_by_tick[agent_type]['human'].append(agent.q_table.get('human', 0.0))
                q_values_by_tick[agent_type]['self_action'].append(agent.q_table.get('self_action', 0.0))
                trust_by_tick[agent_type].append(agent.trust.get('A_0', 0.5))
                prev_info_pending[agent_type] = len(agent.pending_info_evaluations)
                prev_relief_pending[agent_type] = len(agent.pending_rewards)

        # Step
        model.step()

        # Track feedback events that fired this tick
        for agent_type in ['exploratory', 'exploitative']:
            if agent_type in sample_agents:
                agent = sample_agents[agent_type]
                current_info = len(agent.pending_info_evaluations)
                if current_info < prev_info_pending.get(agent_type, 0):
                    info_feedback_counts[agent_type] += (prev_info_pending[agent_type] - current_info)
                    feedback_timeline['info'].append((tick, agent_type))
                current_relief = len(agent.pending_rewards)
                if current_relief < prev_relief_pending.get(agent_type, 0):
                    relief_feedback_counts[agent_type] += (prev_relief_pending[agent_type] - current_relief)
                    feedback_timeline['relief'].append((tick, agent_type))

        # Extract SECI from model (model appends each tick)
        if model.seci_data and len(model.seci_data) > 0:
            latest = model.seci_data[-1]
            seci_by_tick['exploit'].append(latest[1])
            seci_by_tick['explor'].append(latest[2])

        # Extract AECI from model
        if model.aeci_data and len(model.aeci_data) > 0:
            latest = model.aeci_data[-1]
            aeci_by_tick['exploit'].append(latest[1])
            aeci_by_tick['explor'].append(latest[2])

        # Collect MAE and AI preference every 10 ticks
        if tick % 10 == 0:
            for agent_type_label in ['exploratory', 'exploitative']:
                errors = []
                ai_rates = []
                for agent in model.agent_list:
                    if isinstance(agent, HumanAgent) and agent.agent_type == agent_type_label:
                        # MAE
                        mae = 0
                        count = 0
                        for cell, belief_info in agent.beliefs.items():
                            if isinstance(belief_info, dict):
                                belief_level = belief_info.get('level', 0)
                                true_level = model.disaster_grid[cell]
                                mae += abs(belief_level - true_level)
                                count += 1
                        if count > 0:
                            errors.append(mae / count)
                        # AI preference rate
                        if hasattr(agent, 'accum_calls_total') and agent.accum_calls_total > 0:
                            ai_rates.append(agent.accum_calls_ai / agent.accum_calls_total)

                belief_accuracy[agent_type_label].append(np.mean(errors) if errors else 0)
                ai_preference[agent_type_label].append(np.mean(ai_rates) if ai_rates else 0)

    print(f"\nFeedback Event Summary:")
    print(f"  Exploratory  - Info: {info_feedback_counts['exploratory']}, Relief: {relief_feedback_counts['exploratory']}")
    print(f"  Exploitative - Info: {info_feedback_counts['exploitative']}, Relief: {relief_feedback_counts['exploitative']}")
    if seci_by_tick['exploit']:
        print(f"  Final SECI - Exploit: {seci_by_tick['exploit'][-1]:.3f}, Explor: {seci_by_tick['explor'][-1]:.3f}")
    if aeci_by_tick['exploit']:
        print(f"  Final AECI - Exploit: {aeci_by_tick['exploit'][-1]:.3f}, Explor: {aeci_by_tick['explor'][-1]:.3f}")
    if belief_accuracy['exploratory']:
        print(f"  Final MAE  - Exploit: {belief_accuracy['exploitative'][-1]:.3f}, Explor: {belief_accuracy['exploratory'][-1]:.3f}")

    return {
        'q_values': q_values_by_tick,
        'trust': trust_by_tick,
        'info_counts': info_feedback_counts,
        'relief_counts': relief_feedback_counts,
        'feedback_timeline': feedback_timeline,
        'seci': seci_by_tick,
        'aeci': aeci_by_tick,
        'belief_accuracy': belief_accuracy,
        'ai_preference': ai_preference,
        'sample_agents': sample_agents,
        'model': model
    }


def visualize_results(results_high, results_low):
    """Create comprehensive 4x3 visualization with all metrics."""
    fig = plt.figure(figsize=(18, 16))
    fig.suptitle('Dual-Timeline Feedback: High Alignment (0.9) vs Low Alignment (0.1)',
                 fontsize=14, fontweight='bold', y=0.98)

    # === Row 1: Q-values and Trust ===

    # 1. Q-Values (High Alignment)
    ax = plt.subplot(4, 3, 1)
    for source in ['ai', 'human', 'self_action']:
        for agent_type in ['exploratory', 'exploitative']:
            data = results_high['q_values'][agent_type][source]
            ls = '-' if agent_type == 'exploratory' else '--'
            ax.plot(data, linestyle=ls, label=f"{agent_type[:6]}: {source}", alpha=0.8)
    ax.set_title('Q-Values: Confirming AI (0.9)', fontsize=10, fontweight='bold')
    ax.set_xlabel('Tick')
    ax.set_ylabel('Q-Value')
    ax.legend(fontsize=6, loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle=':', alpha=0.3)

    # 2. Q-Values (Low Alignment)
    ax = plt.subplot(4, 3, 2)
    for source in ['ai', 'human', 'self_action']:
        for agent_type in ['exploratory', 'exploitative']:
            data = results_low['q_values'][agent_type][source]
            ls = '-' if agent_type == 'exploratory' else '--'
            ax.plot(data, linestyle=ls, label=f"{agent_type[:6]}: {source}", alpha=0.8)
    ax.set_title('Q-Values: Truthful AI (0.1)', fontsize=10, fontweight='bold')
    ax.set_xlabel('Tick')
    ax.set_ylabel('Q-Value')
    ax.legend(fontsize=6, loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle=':', alpha=0.3)

    # 3. Trust Evolution
    ax = plt.subplot(4, 3, 3)
    ax.plot(results_high['trust']['exploratory'], '-', label='Explor: High (0.9)', color='orange', alpha=0.8)
    ax.plot(results_high['trust']['exploitative'], '--', label='Exploit: High (0.9)', color='orange', alpha=0.8)
    ax.plot(results_low['trust']['exploratory'], '-', label='Explor: Low (0.1)', color='green', alpha=0.8)
    ax.plot(results_low['trust']['exploitative'], '--', label='Exploit: Low (0.1)', color='green', alpha=0.8)
    ax.set_title('Trust in AI (A_0) Evolution', fontsize=10, fontweight='bold')
    ax.set_xlabel('Tick')
    ax.set_ylabel('Trust')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # === Row 2: Feedback Timelines ===

    # 4. Feedback Timeline (High)
    ax = plt.subplot(4, 3, 4)
    for event_type, y_explor, y_exploit, marker_e, marker_x, color_e, color_x in [
        ('info', 1.0, 0.8, 'o', 's', 'blue', 'lightblue'),
        ('relief', 0.4, 0.2, '^', 'v', 'red', 'pink'),
    ]:
        explor = [t for t, at in results_high['feedback_timeline'][event_type] if at == 'exploratory']
        exploit = [t for t, at in results_high['feedback_timeline'][event_type] if at == 'exploitative']
        if explor:
            ax.scatter(explor, [y_explor]*len(explor), marker=marker_e, s=30, alpha=0.6, color=color_e, label=f'{event_type.title()}: Explor')
        if exploit:
            ax.scatter(exploit, [y_exploit]*len(exploit), marker=marker_x, s=30, alpha=0.6, color=color_x, label=f'{event_type.title()}: Exploit')
    ax.set_title('Feedback Timeline: High Align', fontsize=10, fontweight='bold')
    ax.set_xlabel('Tick')
    ax.set_yticks([0.2, 0.4, 0.8, 1.0])
    ax.set_yticklabels(['Relief\nExploit', 'Relief\nExplor', 'Info\nExploit', 'Info\nExplor'], fontsize=7)
    ax.legend(fontsize=6, loc='upper right')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_ylim(0, 1.2)

    # 5. Feedback Timeline (Low)
    ax = plt.subplot(4, 3, 5)
    for event_type, y_explor, y_exploit, marker_e, marker_x, color_e, color_x in [
        ('info', 1.0, 0.8, 'o', 's', 'blue', 'lightblue'),
        ('relief', 0.4, 0.2, '^', 'v', 'red', 'pink'),
    ]:
        explor = [t for t, at in results_low['feedback_timeline'][event_type] if at == 'exploratory']
        exploit = [t for t, at in results_low['feedback_timeline'][event_type] if at == 'exploitative']
        if explor:
            ax.scatter(explor, [y_explor]*len(explor), marker=marker_e, s=30, alpha=0.6, color=color_e, label=f'{event_type.title()}: Explor')
        if exploit:
            ax.scatter(exploit, [y_exploit]*len(exploit), marker=marker_x, s=30, alpha=0.6, color=color_x, label=f'{event_type.title()}: Exploit')
    ax.set_title('Feedback Timeline: Low Align', fontsize=10, fontweight='bold')
    ax.set_xlabel('Tick')
    ax.set_yticks([0.2, 0.4, 0.8, 1.0])
    ax.set_yticklabels(['Relief\nExploit', 'Relief\nExplor', 'Info\nExploit', 'Info\nExplor'], fontsize=7)
    ax.legend(fontsize=6, loc='upper right')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_ylim(0, 1.2)

    # 6. Feedback Frequency Comparison
    ax = plt.subplot(4, 3, 6)
    categories = ['Explor\nInfo', 'Explor\nRelief', 'Exploit\nInfo', 'Exploit\nRelief']
    high_vals = [results_high['info_counts']['exploratory'], results_high['relief_counts']['exploratory'],
                 results_high['info_counts']['exploitative'], results_high['relief_counts']['exploitative']]
    low_vals = [results_low['info_counts']['exploratory'], results_low['relief_counts']['exploratory'],
                results_low['info_counts']['exploitative'], results_low['relief_counts']['exploitative']]
    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width/2, high_vals, width, label='High (0.9)', alpha=0.8, color='orange')
    ax.bar(x + width/2, low_vals, width, label='Low (0.1)', alpha=0.8, color='green')
    ax.set_title('Feedback Event Frequency', fontsize=10, fontweight='bold')
    ax.set_ylabel('Count')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # === Row 3: Echo Chamber Indices ===

    # 7. SECI Evolution
    ax = plt.subplot(4, 3, 7)
    for label, results, color in [('High (0.9)', results_high, 'orange'), ('Low (0.1)', results_low, 'green')]:
        if results['seci']['exploit']:
            ax.plot(results['seci']['exploit'], '--', color=color, alpha=0.8, linewidth=1.5, label=f'{label} Exploit')
            ax.plot(results['seci']['explor'], '-', color=color, alpha=0.8, linewidth=1.5, label=f'{label} Explor')
    ax.axhline(y=0, color='k', linestyle=':', alpha=0.5, label='No echo chamber')
    ax.set_title('SECI: Social Echo Chamber Index\n(Negative = filter bubble)', fontsize=10, fontweight='bold')
    ax.set_xlabel('Tick')
    ax.set_ylabel('SECI (-1 to +1)')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.1, 1.1)

    # 8. AECI Evolution
    ax = plt.subplot(4, 3, 8)
    for label, results, color in [('High (0.9)', results_high, 'orange'), ('Low (0.1)', results_low, 'green')]:
        if results['aeci']['exploit']:
            ax.plot(results['aeci']['exploit'], '--', color=color, alpha=0.8, linewidth=1.5, label=f'{label} Exploit')
            ax.plot(results['aeci']['explor'], '-', color=color, alpha=0.8, linewidth=1.5, label=f'{label} Explor')
    ax.set_title('AECI: AI Echo Chamber Index\n(Higher = more AI reliance)', fontsize=10, fontweight='bold')
    ax.set_xlabel('Tick')
    ax.set_ylabel('AECI (0 to 1)')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)

    # 9. AI vs Friend Preference
    ax = plt.subplot(4, 3, 9)
    for label, results, color in [('High (0.9)', results_high, 'orange'), ('Low (0.1)', results_low, 'green')]:
        for agent_type in ['exploratory', 'exploitative']:
            data = results['ai_preference'][agent_type]
            ticks = list(range(0, len(data) * 10, 10))
            ls = '-' if agent_type == 'exploratory' else '--'
            ax.plot(ticks, data, linestyle=ls, color=color, linewidth=1.5, alpha=0.8,
                    label=f'{label} {agent_type[:6]}')
    ax.axhline(y=0.5, color='k', linestyle=':', alpha=0.5, label='Equal preference')
    ax.set_title('AI vs Friend Preference\n(>0.5 = prefers AI, <0.5 = prefers friends)', fontsize=10, fontweight='bold')
    ax.set_xlabel('Tick')
    ax.set_ylabel('AI Query Rate')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # === Row 4: Belief Accuracy and Summary ===

    # 10. Belief Accuracy (MAE) Evolution
    ax = plt.subplot(4, 3, 10)
    for label, results, color in [('High (0.9)', results_high, 'orange'), ('Low (0.1)', results_low, 'green')]:
        for agent_type in ['exploratory', 'exploitative']:
            data = results['belief_accuracy'][agent_type]
            ticks = list(range(0, len(data) * 10, 10))
            ls = '-' if agent_type == 'exploratory' else '--'
            ax.plot(ticks, data, linestyle=ls, color=color, linewidth=1.5, alpha=0.8,
                    label=f'{label} {agent_type[:6]}')
    ax.set_title('Belief Accuracy (MAE)\n(Lower = more accurate)', fontsize=10, fontweight='bold')
    ax.set_xlabel('Tick')
    ax.set_ylabel('Mean Absolute Error')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

    # 11. SECI vs Normalized AECI overlay
    ax = plt.subplot(4, 3, 11)
    for label, results, color in [('High (0.9)', results_high, 'orange'), ('Low (0.1)', results_low, 'green')]:
        if results['seci']['exploit'] and results['aeci']['exploit']:
            seci = results['seci']['exploit']
            aeci_norm = [2 * (v - 0.5) for v in results['aeci']['exploit']]
            min_len = min(len(seci), len(aeci_norm))
            ax.plot(seci[:min_len], linewidth=2, alpha=0.8, color=color, label=f'{label} SECI')
            ax.plot(aeci_norm[:min_len], linewidth=2, alpha=0.4, color=color, linestyle='--', label=f'{label} AECI (norm)')
    ax.axhline(y=0, color='k', linestyle=':', alpha=0.5)
    ax.set_title('SECI vs Normalized AECI: Exploitative\n(Both on -1 to +1 scale)', fontsize=10, fontweight='bold')
    ax.set_xlabel('Tick')
    ax.set_ylabel('Index Value')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.1, 1.1)

    # 12. Summary
    ax = plt.subplot(4, 3, 12)
    ax.axis('off')

    # Safely get final values
    def safe_last(lst, default=0):
        return lst[-1] if lst else default

    mae_h_expl = safe_last(results_high['belief_accuracy']['exploratory'])
    mae_h_expt = safe_last(results_high['belief_accuracy']['exploitative'])
    mae_l_expl = safe_last(results_low['belief_accuracy']['exploratory'])
    mae_l_expt = safe_last(results_low['belief_accuracy']['exploitative'])
    seci_h = safe_last(results_high['seci']['exploit'])
    seci_l = safe_last(results_low['seci']['exploit'])
    aeci_h = safe_last(results_high['aeci']['exploit'])
    aeci_l = safe_last(results_low['aeci']['exploit'])
    pref_h = safe_last(results_high['ai_preference']['exploitative'])
    pref_l = safe_last(results_low['ai_preference']['exploitative'])

    summary = f"""DUAL-TIMELINE FEEDBACK SUMMARY

High Alignment (0.9) - Confirming AI:
  Info/Relief: Expl={results_high['info_counts']['exploratory']}/{results_high['relief_counts']['exploratory']}, Expt={results_high['info_counts']['exploitative']}/{results_high['relief_counts']['exploitative']}
  SECI: {seci_h:.3f}  AECI: {aeci_h:.3f}
  MAE:  Expl={mae_h_expl:.3f}, Expt={mae_h_expt:.3f}
  AI Pref (Expt): {pref_h:.3f}

Low Alignment (0.1) - Truthful AI:
  Info/Relief: Expl={results_low['info_counts']['exploratory']}/{results_low['relief_counts']['exploratory']}, Expt={results_low['info_counts']['exploitative']}/{results_low['relief_counts']['exploitative']}
  SECI: {seci_l:.3f}  AECI: {aeci_l:.3f}
  MAE:  Expl={mae_l_expl:.3f}, Expt={mae_l_expt:.3f}
  AI Pref (Expt): {pref_l:.3f}

Key Questions Answered:
  Echo chambers? SECI < 0 = yes
  AI reliance?   AECI > 0.5 = high
  Belief quality? Lower MAE = better
  AI preferred?  Pref > 0.5 = yes"""

    ax.text(0.05, 0.95, summary, fontsize=8.5, family='monospace',
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_dir = 'test_results'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'dual_feedback_test.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")

    return fig

if __name__ == "__main__":
    print("="*60)
    print("DUAL-TIMELINE FEEDBACK MECHANISM TEST PROTOCOL")
    print("="*60)

    results_high = run_test(ai_alignment=0.9, test_name="High Alignment (Confirming AI)")
    results_low = run_test(ai_alignment=0.1, test_name="Low Alignment (Truthful AI)")

    print("\nGenerating visualizations...")
    fig = visualize_results(results_high, results_low)

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print("\nExpected Results:")
    print("1. Exploratory agents get MORE info quality feedback (more sensing)")
    print("2. High alignment: Exploratory AI Q-value should DECREASE or stay low")
    print("3. Low alignment: Exploratory AI Q-value should INCREASE")
    print("4. SECI should be more negative under confirming AI")
    print("5. AECI shows AI reliance increasing over time")
    print("6. MAE should be lower under truthful AI (better beliefs)")
    print("7. AI preference diverges between alignment conditions")
