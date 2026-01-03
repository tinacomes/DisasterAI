"""
Trust Dynamics Temporal Analysis - Colab Experiment
====================================================

This notebook diagnoses and analyzes the "sudden trust peak" problem where
trust in AI peaks suddenly for all agent types regardless of alignment.

Run sections in order:
1. Quick Diagnostic - Confirm the problem exists
2. Temporal Evolution - Track trust over time
3. Feedback Analysis - Which mechanism causes peaks?
4. Parameter Tuning - Find optimal learning rates

Author: Claude (DisasterAI debugging session)
Date: 2026-01-03
"""

import numpy as np
import matplotlib.pyplot as plt
from DisasterAI_Model import DisasterModel, HumanAgent
import pandas as pd
from scipy import stats
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

#############################################
# Section 1: QUICK DIAGNOSTIC
#############################################

def run_quick_diagnostic(ai_alignment=0.9, agent_type_filter='exploratory'):
    """
    Quick test to confirm trust peaks early and suddenly.

    Returns:
        dict: Diagnostic results with peak timing, values, and statistics
    """
    print("="*70)
    print("QUICK DIAGNOSTIC: Trust Peak Detection")
    print("="*70)
    print(f"Testing: AI Alignment = {ai_alignment}, Agent Type = {agent_type_filter}")
    print()

    # Setup
    params = {
        'share_exploitative': 0.5,
        'share_of_disaster': 0.15,
        'initial_trust': 0.3,
        'initial_ai_trust': 0.25,
        'number_of_humans': 100,
        'ai_alignment_level': ai_alignment,
        'disaster_dynamics': 2,
        'width': 30,
        'height': 30,
        'ticks': 200,
        'learning_rate': 0.1,
        'epsilon': 0.3,
        'exploit_trust_lr': 0.015,
        'explor_trust_lr': 0.03,
    }

    model = DisasterModel(**params)

    # Filter agents by type
    filtered_agents = [
        agent for agent in model.agent_list
        if isinstance(agent, HumanAgent) and agent.agent_type == agent_type_filter
    ]

    num_agents = len(filtered_agents)
    num_ticks = params['ticks']

    # Track every agent, every tick
    trust_matrix = np.zeros((num_agents, num_ticks))
    velocity_matrix = np.zeros((num_agents, num_ticks - 1))

    print(f"Tracking {num_agents} {agent_type_filter} agents for {num_ticks} ticks...")

    for tick in range(num_ticks):
        model.step()

        # Record all agents' AI trust
        for i, agent in enumerate(filtered_agents):
            trust_matrix[i, tick] = agent.trust.get('A_0', 0.25)

        # Calculate velocity (rate of change)
        if tick > 0:
            velocity_matrix[:, tick - 1] = trust_matrix[:, tick] - trust_matrix[:, tick - 1]

        if (tick + 1) % 50 == 0:
            mean_trust = trust_matrix[:, tick].mean()
            print(f"  Tick {tick + 1}: Mean trust = {mean_trust:.3f}")

    # Analysis
    peak_ticks = np.argmax(trust_matrix, axis=1)
    peak_values = np.max(trust_matrix, axis=1)
    final_values = trust_matrix[:, -1]

    early_peaks = np.sum(peak_ticks < 30)
    max_velocity = np.max(np.abs(velocity_matrix), axis=1).mean()

    # Find when trust first exceeds 0.5
    threshold_ticks = []
    for i in range(num_agents):
        exceeds = np.where(trust_matrix[i, :] > 0.5)[0]
        if len(exceeds) > 0:
            threshold_ticks.append(exceeds[0])
        else:
            threshold_ticks.append(num_ticks)

    print()
    print("="*70)
    print("DIAGNOSTIC RESULTS:")
    print("="*70)
    print(f"Mean peak tick: {peak_ticks.mean():.1f} (±{peak_ticks.std():.1f})")
    print(f"Mean peak value: {peak_values.mean():.3f} (±{peak_values.std():.3f})")
    print(f"Final value: {final_values.mean():.3f} (±{final_values.std():.3f})")
    print(f"Agents peaking before tick 30: {early_peaks}/{num_agents} ({100*early_peaks/num_agents:.1f}%)")
    print(f"Mean tick to exceed 0.5 trust: {np.mean(threshold_ticks):.1f}")
    print(f"Max trust velocity: {max_velocity:.4f} per tick")
    print()

    if peak_ticks.mean() < 30:
        print("⚠️  EARLY PEAK PROBLEM CONFIRMED!")
        print("   Trust peaks very early, likely before sufficient evidence.")
    else:
        print("✅ Peak timing seems reasonable (>30 ticks)")

    if peak_values.mean() > 0.8:
        print("⚠️  CEILING EFFECT CONFIRMED!")
        print("   Trust peaks near maximum, suggesting overshooting.")
    elif peak_values.mean() < 0.3:
        print("⚠️  FLOOR EFFECT CONFIRMED!")
        print("   Trust drops near minimum, suggesting undershooting.")
    else:
        print("✅ Peak values in reasonable range (0.3-0.8)")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1. Trust evolution (sample of 10 agents)
    ax1 = axes[0, 0]
    sample_agents = np.random.choice(num_agents, min(10, num_agents), replace=False)
    for i in sample_agents:
        ax1.plot(trust_matrix[i, :], alpha=0.5, linewidth=1)
    ax1.plot(trust_matrix.mean(axis=0), 'k-', linewidth=2, label='Mean')
    ax1.axhline(0.5, color='r', linestyle='--', alpha=0.3, label='Threshold')
    ax1.set_xlabel('Tick')
    ax1.set_ylabel('Trust in AI')
    ax1.set_title(f'Trust Evolution: {agent_type_filter.title()} Agents\n(AI Alignment = {ai_alignment})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Peak distribution
    ax2 = axes[0, 1]
    ax2.hist(peak_ticks, bins=20, edgecolor='black', alpha=0.7)
    ax2.axvline(peak_ticks.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean = {peak_ticks.mean():.1f}')
    ax2.axvline(30, color='orange', linestyle=':', linewidth=2, label='Tick 30 (concern threshold)')
    ax2.set_xlabel('Tick of Peak Trust')
    ax2.set_ylabel('Number of Agents')
    ax2.set_title('Distribution of Peak Timing')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Trust velocity (heatmap)
    ax3 = axes[1, 0]
    # Show mean velocity over time
    mean_velocity = velocity_matrix.mean(axis=0)
    ax3.plot(mean_velocity, linewidth=2)
    ax3.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax3.fill_between(range(len(mean_velocity)),
                     mean_velocity - velocity_matrix.std(axis=0),
                     mean_velocity + velocity_matrix.std(axis=0),
                     alpha=0.3)
    ax3.set_xlabel('Tick')
    ax3.set_ylabel('Trust Velocity (Δtrust/tick)')
    ax3.set_title('Rate of Trust Change Over Time')
    ax3.grid(True, alpha=0.3)

    # 4. Peak value vs peak timing
    ax4 = axes[1, 1]
    scatter = ax4.scatter(peak_ticks, peak_values, alpha=0.5, s=30)
    ax4.set_xlabel('Tick of Peak')
    ax4.set_ylabel('Peak Trust Value')
    ax4.set_title('Peak Timing vs. Peak Magnitude')
    ax4.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(peak_ticks, peak_values, 1)
    p = np.poly1d(z)
    ax4.plot(peak_ticks, p(peak_ticks), "r--", alpha=0.8, linewidth=2)

    plt.tight_layout()
    plt.savefig(f'diagnostic_trust_peaks_{agent_type_filter}_align{ai_alignment}.png', dpi=150, bbox_inches='tight')
    plt.show()

    return {
        'trust_matrix': trust_matrix,
        'velocity_matrix': velocity_matrix,
        'peak_ticks': peak_ticks,
        'peak_values': peak_values,
        'final_values': final_values,
        'threshold_ticks': threshold_ticks,
        'params': params
    }


#############################################
# Section 2: TEMPORAL EVOLUTION ANALYSIS
#############################################

def run_temporal_evolution_experiment():
    """
    Full 2×4 experimental design tracking trust evolution.

    Tests all combinations of:
    - Agent types: exploratory, exploitative
    - AI alignments: None (control), 0.1 (truthful), 0.5 (mixed), 0.9 (confirming)

    Returns:
        DataFrame with temporal metrics for all conditions
    """
    print("="*70)
    print("TEMPORAL EVOLUTION EXPERIMENT")
    print("="*70)
    print("Testing 2 agent types × 4 alignment levels = 8 conditions")
    print()

    agent_types = ['exploratory', 'exploitative']
    ai_alignments = [None, 0.1, 0.5, 0.9]
    alignment_names = {None: 'Control', 0.1: 'Truthful', 0.5: 'Mixed', 0.9: 'Confirming'}

    base_params = {
        'share_exploitative': 0.5,
        'share_of_disaster': 0.15,
        'initial_trust': 0.3,
        'initial_ai_trust': 0.25,
        'number_of_humans': 100,
        'disaster_dynamics': 2,
        'width': 30,
        'height': 30,
        'ticks': 200,
        'learning_rate': 0.1,
        'epsilon': 0.3,
        'exploit_trust_lr': 0.015,
        'explor_trust_lr': 0.03,
    }

    results = []

    for alignment in ai_alignments:
        print(f"\n{'='*70}")
        print(f"Running: AI Alignment = {alignment_names[alignment]}")
        print(f"{'='*70}\n")

        params = base_params.copy()
        if alignment is not None:
            params['ai_alignment_level'] = alignment
        else:
            params['ai_alignment_level'] = 0.5  # Dummy (will remove AI)

        model = DisasterModel(**params)

        # For control, remove AI
        if alignment is None:
            model.num_ai = 0
            model.ai_list = []
            print("Control condition: AI agents removed\n")

        # Separate agents by type
        agents_by_type = {'exploratory': [], 'exploitative': []}
        for agent in model.agent_list:
            if isinstance(agent, HumanAgent):
                agents_by_type[agent.agent_type].append(agent)

        # Track metrics per tick
        temporal_data = {agent_type: {
            'tick': [],
            'trust_mean': [],
            'trust_std': [],
            'trust_q25': [],
            'trust_q75': [],
            'trust_min': [],
            'trust_max': [],
            'velocity': [],
            'ai_usage_rate': []
        } for agent_type in agent_types}

        # Run simulation
        prev_trust = {agent_type: None for agent_type in agent_types}

        for tick in range(params['ticks']):
            model.step()

            for agent_type in agent_types:
                agents = agents_by_type[agent_type]
                if len(agents) == 0:
                    continue

                # Collect trust values
                trust_values = [agent.trust.get('A_0', 0.25) for agent in agents]

                # Compute statistics
                temporal_data[agent_type]['tick'].append(tick)
                temporal_data[agent_type]['trust_mean'].append(np.mean(trust_values))
                temporal_data[agent_type]['trust_std'].append(np.std(trust_values))
                temporal_data[agent_type]['trust_q25'].append(np.percentile(trust_values, 25))
                temporal_data[agent_type]['trust_q75'].append(np.percentile(trust_values, 75))
                temporal_data[agent_type]['trust_min'].append(np.min(trust_values))
                temporal_data[agent_type]['trust_max'].append(np.max(trust_values))

                # AI usage rate
                if alignment is not None:
                    ai_users = sum(1 for agent in agents if agent.accepted_ai > 0)
                    temporal_data[agent_type]['ai_usage_rate'].append(ai_users / len(agents))
                else:
                    temporal_data[agent_type]['ai_usage_rate'].append(0)

                # Velocity (rate of change)
                if prev_trust[agent_type] is not None:
                    velocity = np.mean(trust_values) - prev_trust[agent_type]
                    temporal_data[agent_type]['velocity'].append(velocity)
                else:
                    temporal_data[agent_type]['velocity'].append(0)

                prev_trust[agent_type] = np.mean(trust_values)

            if (tick + 1) % 50 == 0:
                print(f"  Tick {tick + 1} complete")

        # Store results
        for agent_type in agent_types:
            result_row = {
                'alignment': alignment_names[alignment],
                'alignment_value': alignment if alignment is not None else -1,
                'agent_type': agent_type,
                'temporal_data': temporal_data[agent_type]
            }

            # Compute summary statistics
            trust_mean = temporal_data[agent_type]['trust_mean']
            if len(trust_mean) > 0:
                result_row['peak_tick'] = np.argmax(trust_mean)
                result_row['peak_value'] = np.max(trust_mean)
                result_row['final_value'] = trust_mean[-1]
                result_row['initial_value'] = trust_mean[0]
                result_row['mean_value'] = np.mean(trust_mean)

                # Time to reach 50% of peak
                half_peak = result_row['peak_value'] / 2
                exceeds = np.where(np.array(trust_mean) > half_peak)[0]
                result_row['half_life'] = exceeds[0] if len(exceeds) > 0 else params['ticks']

                # Max velocity
                velocities = temporal_data[agent_type]['velocity']
                result_row['max_velocity'] = np.max(np.abs(velocities)) if len(velocities) > 0 else 0

            results.append(result_row)

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)

    return pd.DataFrame(results)


def visualize_temporal_evolution(results_df):
    """Create comprehensive visualization of temporal evolution."""
    print("\nGenerating temporal evolution visualizations...")

    agent_types = results_df['agent_type'].unique()
    alignments = results_df['alignment'].unique()

    # Create 2×4 grid
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    colors = {'Control': 'gray', 'Truthful': 'green', 'Mixed': 'orange', 'Confirming': 'red'}

    for i, agent_type in enumerate(['exploratory', 'exploitative']):
        for j, alignment in enumerate(['Control', 'Truthful', 'Mixed', 'Confirming']):
            ax = axes[i, j]

            # Get data for this condition
            row = results_df[(results_df['agent_type'] == agent_type) &
                            (results_df['alignment'] == alignment)]

            if len(row) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                continue

            row = row.iloc[0]
            temporal = row['temporal_data']

            ticks = temporal['tick']
            mean_trust = temporal['trust_mean']
            q25 = temporal['trust_q25']
            q75 = temporal['trust_q75']

            # Plot mean with IQR band
            ax.plot(ticks, mean_trust, color=colors[alignment], linewidth=2, label='Mean')
            ax.fill_between(ticks, q25, q75, alpha=0.3, color=colors[alignment])

            # Mark peak
            if 'peak_tick' in row:
                ax.axvline(row['peak_tick'], color='red', linestyle='--', alpha=0.5, linewidth=1)
                ax.scatter([row['peak_tick']], [row['peak_value']], color='red', s=100, zorder=5, marker='*')

            # Formatting
            ax.set_xlim(0, 200)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)

            if i == 0:
                ax.set_title(f'{alignment}\n({agent_type.title()})', fontweight='bold')
            else:
                ax.set_title(f'{agent_type.title()}', fontweight='bold')

            if j == 0:
                ax.set_ylabel('Trust in AI')
            if i == 1:
                ax.set_xlabel('Tick')

            # Add summary stats as text
            if 'peak_tick' in row:
                stats_text = f"Peak: t={row['peak_tick']:.0f}, v={row['peak_value']:.2f}\nFinal: {row['final_value']:.2f}"
                ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
                       fontsize=8, verticalalignment='bottom', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Trust in AI: Temporal Evolution by Agent Type and AI Alignment',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('temporal_evolution_full.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Second figure: Velocity analysis
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))

    for i, agent_type in enumerate(['exploratory', 'exploitative']):
        ax = axes2[i]

        for alignment in ['Control', 'Truthful', 'Mixed', 'Confirming']:
            row = results_df[(results_df['agent_type'] == agent_type) &
                            (results_df['alignment'] == alignment)]

            if len(row) == 0:
                continue

            row = row.iloc[0]
            temporal = row['temporal_data']

            velocities = temporal['velocity']
            ax.plot(velocities, color=colors[alignment], linewidth=2, label=alignment, alpha=0.8)

        ax.axhline(0, color='k', linestyle='-', alpha=0.3)
        ax.set_xlabel('Tick')
        ax.set_ylabel('Trust Velocity (Δtrust/tick)')
        ax.set_title(f'Rate of Trust Change: {agent_type.title()} Agents', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('trust_velocity_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("✅ Visualizations saved:")
    print("   - temporal_evolution_full.png")
    print("   - trust_velocity_analysis.png")


#############################################
# Section 3: SUMMARY STATISTICS
#############################################

def print_summary_statistics(results_df):
    """Print summary statistics and hypothesis tests."""
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    # Group by agent type and alignment
    summary = results_df.groupby(['agent_type', 'alignment']).agg({
        'peak_tick': ['mean', 'std'],
        'peak_value': ['mean', 'std'],
        'final_value': ['mean', 'std'],
        'half_life': ['mean', 'std'],
        'max_velocity': ['mean', 'std']
    }).round(3)

    print("\n", summary)

    # Hypothesis tests
    print("\n" + "="*70)
    print("HYPOTHESIS TESTS")
    print("="*70)

    # H1: Exploratory + Confirming shows early peak
    explor_confirm = results_df[(results_df['agent_type'] == 'exploratory') &
                                (results_df['alignment'] == 'Confirming')]
    explor_truthful = results_df[(results_df['agent_type'] == 'exploratory') &
                                 (results_df['alignment'] == 'Truthful')]

    if len(explor_confirm) > 0 and len(explor_truthful) > 0:
        t_stat, p_value = stats.ttest_ind(
            [explor_confirm.iloc[0]['peak_tick']],
            [explor_truthful.iloc[0]['peak_tick']]
        )
        print(f"\nH1: Exploratory agents peak earlier with Confirming vs Truthful AI")
        print(f"   Confirming peak: {explor_confirm.iloc[0]['peak_tick']:.1f}")
        print(f"   Truthful peak: {explor_truthful.iloc[0]['peak_tick']:.1f}")
        print(f"   Difference: {explor_confirm.iloc[0]['peak_tick'] - explor_truthful.iloc[0]['peak_tick']:.1f} ticks")

    # H2: Final trust differs by alignment for exploratory agents
    explor_data = results_df[results_df['agent_type'] == 'exploratory']
    print(f"\nH2: Exploratory agents' final trust varies by AI alignment")
    for _, row in explor_data.iterrows():
        if row['alignment'] != 'Control':
            print(f"   {row['alignment']}: {row['final_value']:.3f}")

    # Expected pattern: Truthful > Mixed > Confirming

    print("\n" + "="*70)


#############################################
# Section 4: MAIN EXECUTION
#############################################

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║   Trust Dynamics Temporal Analysis - DisasterAI Debugging        ║
    ║                                                                   ║
    ║   This experiment diagnoses sudden trust peaks and tracks         ║
    ║   temporal evolution instead of final values.                     ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)

    # Step 1: Quick Diagnostic
    print("\n" + "="*70)
    print("STEP 1: Quick Diagnostic")
    print("="*70)
    print("Running quick test to confirm trust peak problem...")
    print()

    diagnostic_results = run_quick_diagnostic(
        ai_alignment=0.9,
        agent_type_filter='exploratory'
    )

    input("\nPress Enter to continue to full temporal evolution experiment...")

    # Step 2: Full Temporal Evolution
    print("\n" + "="*70)
    print("STEP 2: Full Temporal Evolution Experiment")
    print("="*70)

    results_df = run_temporal_evolution_experiment()

    # Step 3: Visualizations
    visualize_temporal_evolution(results_df)

    # Step 4: Summary Statistics
    print_summary_statistics(results_df)

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE!")
    print("="*70)
    print("\nResults saved:")
    print("  - diagnostic_trust_peaks_exploratory_align0.9.png")
    print("  - temporal_evolution_full.png")
    print("  - trust_velocity_analysis.png")
    print("\nNext steps:")
    print("  1. Review peak timing - are most agents peaking before tick 30?")
    print("  2. Check velocity plots - are there sudden spikes?")
    print("  3. Compare exploratory vs exploitative responses to alignment")
    print("  4. If problems confirmed, run parameter tuning experiments")
