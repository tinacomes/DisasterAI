"""
Diagnostic test: Track WHO agents query at different alignment levels.
This tests whether Q-learning is adapting source preferences correctly.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DisasterAI_Model import DisasterModel
import warnings
warnings.filterwarnings('ignore')

# Calibrated values
CALIBRATED_PARAMS = {
    'exploit_D': 1.5,
    'exploit_delta': 25,
    'explore_D': 2.5,
    'explore_delta': 10,
}

BASE_PARAMS = {
    'share_exploitative': 0.5,
    'share_of_disaster': 0.3,
    'initial_trust': 0.5,
    'initial_ai_trust': 0.5,
    'number_of_humans': 50,
    'share_confirming': 0.5,
    'ticks': 50,
    'memory_exploit': 3,
    'memory_explore': 7,
}


def run_diagnostic(params, seed=None):
    """Run simulation and track source selection behavior."""
    if seed is not None:
        np.random.seed(seed)
        import random
        random.seed(seed)

    model = DisasterModel(**params)

    for _ in range(params['ticks']):
        model.step()

    # Collect detailed metrics by agent type
    exploit_agents = [a for a in model.humans.values() if a.agent_type == 'exploitative']
    explore_agents = [a for a in model.humans.values() if a.agent_type == 'exploratory']

    def get_agent_metrics(agents, agent_label):
        """Get detailed metrics for a group of agents."""
        metrics = {}

        # Q-values for source selection
        q_ai = np.mean([a.q_table.get('ai', 0) for a in agents])
        q_human = np.mean([a.q_table.get('human', 0) for a in agents])
        q_self = np.mean([a.q_table.get('self_action', 0) for a in agents])

        metrics[f'{agent_label}_q_ai'] = q_ai
        metrics[f'{agent_label}_q_human'] = q_human
        metrics[f'{agent_label}_q_self'] = q_self
        metrics[f'{agent_label}_q_ai_minus_human'] = q_ai - q_human  # Positive = prefers AI

        # Trust levels
        ai_trust = []
        human_trust = []
        for a in agents:
            for source, trust in a.trust.items():
                if source.startswith('A_'):
                    ai_trust.append(trust)
                elif source.startswith('H_'):
                    human_trust.append(trust)

        metrics[f'{agent_label}_trust_ai'] = np.mean(ai_trust) if ai_trust else 0
        metrics[f'{agent_label}_trust_human'] = np.mean(human_trust) if human_trust else 0

        # MAE
        errors = []
        for agent in agents:
            for cell, belief in agent.beliefs.items():
                x, y = cell
                actual = model.disaster_grid[x, y]
                errors.append(abs(belief['level'] - actual))
        metrics[f'{agent_label}_mae'] = np.mean(errors) if errors else 0

        # Belief confidence
        confidences = []
        for agent in agents:
            for belief in agent.beliefs.values():
                confidences.append(belief.get('confidence', 0.5))
        metrics[f'{agent_label}_confidence'] = np.mean(confidences) if confidences else 0

        return metrics

    results = {}
    results.update(get_agent_metrics(exploit_agents, 'exploit'))
    results.update(get_agent_metrics(explore_agents, 'explore'))

    return results


def main():
    print("="*60)
    print("DIAGNOSTIC: SOURCE SELECTION BY ALIGNMENT")
    print("="*60)

    alignment_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    n_reps = 5
    results = []

    for align in alignment_levels:
        print(f"\nAlignment = {align}")

        for rep in range(n_reps):
            params = BASE_PARAMS.copy()
            params.update(CALIBRATED_PARAMS)
            params['ai_alignment_level'] = align

            try:
                metrics = run_diagnostic(params, seed=rep*100 + int(align*10))
                metrics['alignment'] = align
                metrics['rep'] = rep
                results.append(metrics)
            except Exception as e:
                print(f"  Rep {rep}: ERROR - {e}")

    df = pd.DataFrame(results)
    df.to_csv('calibration_results/diagnostic_source_selection.csv', index=False)

    # Print summary
    print("\n" + "="*60)
    print("Q-VALUE ANALYSIS (positive = prefers AI over humans)")
    print("="*60)

    summary = df.groupby('alignment').agg({
        'exploit_q_ai_minus_human': 'mean',
        'explore_q_ai_minus_human': 'mean',
        'exploit_trust_ai': 'mean',
        'explore_trust_ai': 'mean',
        'exploit_mae': 'mean',
        'explore_mae': 'mean',
    }).round(4)
    print(summary)

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    agg = df.groupby('alignment').mean()

    # 1. Q-value preference (AI - Human)
    ax = axes[0, 0]
    ax.plot(agg.index, agg['exploit_q_ai_minus_human'], 'o-', color='coral', label='Exploitative', linewidth=2)
    ax.plot(agg.index, agg['explore_q_ai_minus_human'], 's-', color='steelblue', label='Exploratory', linewidth=2)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('AI Alignment Level')
    ax.set_ylabel('Q(AI) - Q(Human)')
    ax.set_title('Source Preference\n(Positive = Prefers AI)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Q-values breakdown - Exploitative
    ax = axes[0, 1]
    ax.plot(agg.index, agg['exploit_q_ai'], 'o-', label='Q(AI)', linewidth=2)
    ax.plot(agg.index, agg['exploit_q_human'], 's-', label='Q(Human)', linewidth=2)
    ax.plot(agg.index, agg['exploit_q_self'], '^-', label='Q(Self)', linewidth=2)
    ax.set_xlabel('AI Alignment Level')
    ax.set_ylabel('Q-Value')
    ax.set_title('Exploitative: Q-Values by Source')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Q-values breakdown - Exploratory
    ax = axes[0, 2]
    ax.plot(agg.index, agg['explore_q_ai'], 'o-', label='Q(AI)', linewidth=2)
    ax.plot(agg.index, agg['explore_q_human'], 's-', label='Q(Human)', linewidth=2)
    ax.plot(agg.index, agg['explore_q_self'], '^-', label='Q(Self)', linewidth=2)
    ax.set_xlabel('AI Alignment Level')
    ax.set_ylabel('Q-Value')
    ax.set_title('Exploratory: Q-Values by Source')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Trust in AI
    ax = axes[1, 0]
    ax.plot(agg.index, agg['exploit_trust_ai'], 'o-', color='coral', label='Exploitative', linewidth=2)
    ax.plot(agg.index, agg['explore_trust_ai'], 's-', color='steelblue', label='Exploratory', linewidth=2)
    ax.set_xlabel('AI Alignment Level')
    ax.set_ylabel('Trust in AI')
    ax.set_title('AI Trust by Alignment')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. MAE (for reference)
    ax = axes[1, 1]
    ax.plot(agg.index, agg['exploit_mae'], 'o-', color='coral', label='Exploitative', linewidth=2)
    ax.plot(agg.index, agg['explore_mae'], 's-', color='steelblue', label='Exploratory', linewidth=2)
    ax.set_xlabel('AI Alignment Level')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Belief Error by Alignment')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Trust in Humans
    ax = axes[1, 2]
    ax.plot(agg.index, agg['exploit_trust_human'], 'o-', color='coral', label='Exploitative', linewidth=2)
    ax.plot(agg.index, agg['explore_trust_human'], 's-', color='steelblue', label='Exploratory', linewidth=2)
    ax.set_xlabel('AI Alignment Level')
    ax.set_ylabel('Trust in Humans')
    ax.set_title('Human Trust by Alignment')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('calibration_results/diagnostic_source_selection.png', dpi=150)
    print(f"\nPlot saved: calibration_results/diagnostic_source_selection.png")
    plt.close()

    # Key findings
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)

    low = df[df['alignment'] == 0.1]
    high = df[df['alignment'] == 0.9]

    print("\nExploitative agents:")
    print(f"  At align=0.1: Q(AI)-Q(Human) = {low['exploit_q_ai_minus_human'].mean():.4f}")
    print(f"  At align=0.9: Q(AI)-Q(Human) = {high['exploit_q_ai_minus_human'].mean():.4f}")
    print(f"  → {'Shifts toward AI' if high['exploit_q_ai_minus_human'].mean() > low['exploit_q_ai_minus_human'].mean() else 'Does NOT shift toward AI'}")

    print("\nExploratory agents:")
    print(f"  At align=0.1: Q(AI)-Q(Human) = {low['explore_q_ai_minus_human'].mean():.4f}")
    print(f"  At align=0.9: Q(AI)-Q(Human) = {high['explore_q_ai_minus_human'].mean():.4f}")
    print(f"  → {'Shifts toward AI' if high['explore_q_ai_minus_human'].mean() > low['explore_q_ai_minus_human'].mean() else 'Does NOT shift toward AI'}")


if __name__ == '__main__':
    main()
