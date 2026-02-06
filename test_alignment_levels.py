"""
Test calibrated D/δ values at different AI alignment levels.
Run: python test_alignment_levels.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DisasterAI_Model import DisasterModel
import warnings
warnings.filterwarnings('ignore')

# Calibrated values from Phase 2 results
CALIBRATED_PARAMS = {
    'exploit_D': 1.5,
    'exploit_delta': 25,
    'explore_D': 2.5,
    'explore_delta': 10,
}

# Base simulation parameters
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

def run_simulation(params, seed=None):
    """Run a single simulation and collect metrics."""
    if seed is not None:
        np.random.seed(seed)
        import random
        random.seed(seed)

    model = DisasterModel(**params)

    for _ in range(params['ticks']):
        model.step()

    # Collect metrics by agent type
    exploit_agents = [a for a in model.humans.values() if a.agent_type == 'exploitative']
    explore_agents = [a for a in model.humans.values() if a.agent_type == 'exploratory']

    # Calculate MAE for each agent type
    def calc_mae(agents):
        errors = []
        for agent in agents:
            for cell, belief in agent.beliefs.items():
                actual = model.grid_levels.get(cell, 0)
                errors.append(abs(belief['level'] - actual))
        return np.mean(errors) if errors else 0

    # Calculate SECI
    def calc_seci(agents):
        if len(agents) < 2:
            return 0
        beliefs_flat = []
        for agent in agents:
            if agent.beliefs:
                beliefs_flat.append(np.mean([b['level'] for b in agent.beliefs.values()]))
        if len(beliefs_flat) < 2:
            return 0
        return -np.var(beliefs_flat)  # Negative variance = echo chamber measure

    return {
        'mae_exploit': calc_mae(exploit_agents),
        'mae_explore': calc_mae(explore_agents),
        'seci_exploit': calc_seci(exploit_agents),
        'seci_explore': calc_seci(explore_agents),
    }


def main():
    print("="*60)
    print("TESTING CALIBRATED D/δ AT DIFFERENT ALIGNMENT LEVELS")
    print("="*60)
    print(f"\nCalibrated parameters:")
    print(f"  Exploitative: D={CALIBRATED_PARAMS['exploit_D']}, δ={CALIBRATED_PARAMS['exploit_delta']}")
    print(f"  Exploratory:  D={CALIBRATED_PARAMS['explore_D']}, δ={CALIBRATED_PARAMS['explore_delta']}")

    alignment_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    n_reps = 3
    results = []

    for align in alignment_levels:
        print(f"\nTesting alignment = {align}...")

        for rep in range(n_reps):
            params = BASE_PARAMS.copy()
            params.update(CALIBRATED_PARAMS)
            params['ai_alignment_level'] = align

            try:
                metrics = run_simulation(params, seed=rep*100 + int(align*10))
                metrics['alignment'] = align
                metrics['replication'] = rep
                results.append(metrics)
                print(f"  Rep {rep+1}: MAE_exploit={metrics['mae_exploit']:.4f}, MAE_explore={metrics['mae_explore']:.4f}")
            except Exception as e:
                print(f"  Rep {rep+1}: ERROR - {e}")

    df = pd.DataFrame(results)
    df.to_csv('calibration_results/alignment_test.csv', index=False)

    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    summary = df.groupby('alignment').agg({
        'mae_exploit': ['mean', 'std'],
        'mae_explore': ['mean', 'std'],
        'seci_exploit': ['mean', 'std'],
        'seci_explore': ['mean', 'std'],
    }).round(4)
    print(summary)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # MAE by alignment
    ax = axes[0, 0]
    agg = df.groupby('alignment').agg({'mae_exploit': 'mean', 'mae_explore': 'mean'})
    ax.plot(agg.index, agg['mae_exploit'], marker='o', label='Exploitative', color='coral', linewidth=2)
    ax.plot(agg.index, agg['mae_explore'], marker='s', label='Exploratory', color='steelblue', linewidth=2)
    ax.set_xlabel('AI Alignment Level')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Belief Accuracy by AI Alignment')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # SECI by alignment
    ax = axes[0, 1]
    agg = df.groupby('alignment').agg({'seci_exploit': 'mean', 'seci_explore': 'mean'})
    ax.plot(agg.index, agg['seci_exploit'], marker='o', label='Exploitative', color='coral', linewidth=2)
    ax.plot(agg.index, agg['seci_explore'], marker='s', label='Exploratory', color='steelblue', linewidth=2)
    ax.set_xlabel('AI Alignment Level')
    ax.set_ylabel('SECI (Social Echo Chamber Index)')
    ax.set_title('Echo Chamber Effect by AI Alignment')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # MAE difference (exploit - explore)
    ax = axes[1, 0]
    agg = df.groupby('alignment').agg({'mae_exploit': 'mean', 'mae_explore': 'mean'})
    diff = agg['mae_exploit'] - agg['mae_explore']
    colors = ['coral' if d > 0 else 'steelblue' for d in diff]
    ax.bar(agg.index, diff, color=colors, width=0.15)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('AI Alignment Level')
    ax.set_ylabel('MAE Difference (Exploit - Explore)')
    ax.set_title('Who Has Better Beliefs?\n(Positive = Explorers better)')
    ax.grid(True, alpha=0.3)

    # Box plot of MAE
    ax = axes[1, 1]
    positions_exploit = [i - 0.1 for i in range(len(alignment_levels))]
    positions_explore = [i + 0.1 for i in range(len(alignment_levels))]

    bp1 = ax.boxplot([df[df['alignment'] == a]['mae_exploit'].values for a in alignment_levels],
                     positions=positions_exploit, widths=0.15, patch_artist=True)
    bp2 = ax.boxplot([df[df['alignment'] == a]['mae_explore'].values for a in alignment_levels],
                     positions=positions_explore, widths=0.15, patch_artist=True)

    for patch in bp1['boxes']:
        patch.set_facecolor('coral')
    for patch in bp2['boxes']:
        patch.set_facecolor('steelblue')

    ax.set_xticks(range(len(alignment_levels)))
    ax.set_xticklabels(alignment_levels)
    ax.set_xlabel('AI Alignment Level')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('MAE Distribution by Alignment')
    ax.legend([bp1['boxes'][0], bp2['boxes'][0]], ['Exploitative', 'Exploratory'])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('calibration_results/alignment_test.png', dpi=150)
    print(f"\nPlot saved: calibration_results/alignment_test.png")
    plt.close()

    # Key findings
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)

    low_align = df[df['alignment'] == 0.1]['mae_exploit'].mean()
    high_align = df[df['alignment'] == 0.9]['mae_exploit'].mean()
    print(f"\nExploitative agents:")
    print(f"  MAE at low alignment (0.1):  {low_align:.4f}")
    print(f"  MAE at high alignment (0.9): {high_align:.4f}")
    print(f"  Change: {((high_align - low_align) / low_align * 100):+.1f}%")

    low_align = df[df['alignment'] == 0.1]['mae_explore'].mean()
    high_align = df[df['alignment'] == 0.9]['mae_explore'].mean()
    print(f"\nExploratory agents:")
    print(f"  MAE at low alignment (0.1):  {low_align:.4f}")
    print(f"  MAE at high alignment (0.9): {high_align:.4f}")
    print(f"  Change: {((high_align - low_align) / low_align * 100):+.1f}%")


if __name__ == '__main__':
    main()
