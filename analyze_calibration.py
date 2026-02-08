"""
Analyze calibration results and create clear visualizations.
Run: python analyze_calibration.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

RESULTS_DIR = 'calibration_results'

def analyze_phase2():
    """Analyze D/δ calibration results."""
    filepath = f'{RESULTS_DIR}/phase2_d_delta.csv'
    if not os.path.exists(filepath):
        print("Phase 2 results not found")
        return

    df = pd.read_csv(filepath)
    print("\n" + "="*60)
    print("PHASE 2: D/δ CALIBRATION ANALYSIS")
    print("="*60)

    print(f"\nData columns: {list(df.columns)}")
    print(f"Agent types found: {df['agent_type'].unique()}")

    # Separate exploiter and explorer results (using actual column values)
    exploit_df = df[df['agent_type'] == 'exploitative']
    explore_df = df[df['agent_type'] == 'exploratory']

    print("\n📊 EXPLOITATIVE Agent Results:")
    print("-" * 40)
    if len(exploit_df) > 0:
        # Group by D and delta, show key metrics
        summary = exploit_df.groupby(['D', 'delta']).agg({
            'acceptance_rate': 'mean',
            'final_mae': 'mean',
            'final_seci': 'mean'
        }).round(4)
        print(summary)

        # Best config
        best_idx = exploit_df.groupby(['D', 'delta'])['final_mae'].mean().idxmin()
        print(f"\n✓ Best exploitative config (lowest MAE): D={best_idx[0]}, δ={best_idx[1]}")

    print("\n📊 EXPLORATORY Agent Results:")
    print("-" * 40)
    if len(explore_df) > 0:
        summary = explore_df.groupby(['D', 'delta']).agg({
            'acceptance_rate': 'mean',
            'final_mae': 'mean',
            'final_seci': 'mean'
        }).round(4)
        print(summary)

        best_idx = explore_df.groupby(['D', 'delta'])['final_mae'].mean().idxmin()
        print(f"\n✓ Best exploratory config (lowest MAE): D={best_idx[0]}, δ={best_idx[1]}")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, (agent_type, agent_df, label) in enumerate([
        ('exploitative', exploit_df, 'Exploitative'),
        ('exploratory', explore_df, 'Exploratory')
    ]):
        if len(agent_df) == 0:
            axes[idx, 0].text(0.5, 0.5, f'No {label} data', ha='center', va='center')
            axes[idx, 1].text(0.5, 0.5, f'No {label} data', ha='center', va='center')
            continue

        # Acceptance rate by D
        ax = axes[idx, 0]
        for delta in sorted(agent_df['delta'].unique()):
            subset = agent_df[agent_df['delta'] == delta].groupby('D')['acceptance_rate'].mean()
            ax.plot(subset.index, subset.values, marker='o', label=f'δ={delta}')
        ax.set_xlabel('D (latitude of acceptance)')
        ax.set_ylabel('Acceptance Rate')
        ax.set_title(f'{label}: Acceptance Rate by D')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # MAE by D
        ax = axes[idx, 1]
        for delta in sorted(agent_df['delta'].unique()):
            subset = agent_df[agent_df['delta'] == delta].groupby('D')['final_mae'].mean()
            ax.plot(subset.index, subset.values, marker='s', label=f'δ={delta}')
        ax.set_xlabel('D (latitude of acceptance)')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title(f'{label}: Belief Error by D')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/analysis_phase2.png', dpi=150)
    print(f"\n📈 Plot saved: {RESULTS_DIR}/analysis_phase2.png")
    plt.close()


def analyze_phase3():
    """Analyze memory size calibration."""
    filepath = f'{RESULTS_DIR}/phase3_memory.csv'
    if not os.path.exists(filepath):
        print("\nPhase 3 results not found")
        return

    df = pd.read_csv(filepath)
    print("\n" + "="*60)
    print("PHASE 3: MEMORY SIZE ANALYSIS")
    print("="*60)

    summary = df.groupby(['memory_exploit', 'memory_explore']).agg({
        'mae_exploit': 'mean',
        'mae_explore': 'mean',
        'seci_exploit': 'mean',
        'seci_explore': 'mean'
    }).round(4)
    print(summary)

    # Best config
    df['total_mae'] = df['mae_exploit'] + df['mae_explore']
    best = df.groupby(['memory_exploit', 'memory_explore'])['total_mae'].mean().idxmin()
    print(f"\n✓ Best memory config: exploit={best[0]}, explore={best[1]}")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Heatmap for exploiter MAE
    pivot = df.pivot_table(values='mae_exploit', index='memory_exploit',
                           columns='memory_explore', aggfunc='mean')
    im = axes[0].imshow(pivot.values, cmap='RdYlGn_r', aspect='auto')
    axes[0].set_xticks(range(len(pivot.columns)))
    axes[0].set_xticklabels(pivot.columns)
    axes[0].set_yticks(range(len(pivot.index)))
    axes[0].set_yticklabels(pivot.index)
    axes[0].set_xlabel('Explorer Memory Size')
    axes[0].set_ylabel('Exploiter Memory Size')
    axes[0].set_title('Exploiter MAE (lower=better)')
    plt.colorbar(im, ax=axes[0])

    # Heatmap for explorer MAE
    pivot = df.pivot_table(values='mae_explore', index='memory_exploit',
                           columns='memory_explore', aggfunc='mean')
    im = axes[1].imshow(pivot.values, cmap='RdYlGn_r', aspect='auto')
    axes[1].set_xticks(range(len(pivot.columns)))
    axes[1].set_xticklabels(pivot.columns)
    axes[1].set_yticks(range(len(pivot.index)))
    axes[1].set_yticklabels(pivot.index)
    axes[1].set_xlabel('Explorer Memory Size')
    axes[1].set_ylabel('Exploiter Memory Size')
    axes[1].set_title('Explorer MAE (lower=better)')
    plt.colorbar(im, ax=axes[1])

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/analysis_phase3.png', dpi=150)
    print(f"📈 Plot saved: {RESULTS_DIR}/analysis_phase3.png")
    plt.close()


def analyze_phase6():
    """Analyze validation results."""
    filepath = f'{RESULTS_DIR}/phase6_validation.csv'
    if not os.path.exists(filepath):
        print("\nPhase 6 results not found")
        return

    df = pd.read_csv(filepath)
    print("\n" + "="*60)
    print("PHASE 6: VALIDATION ANALYSIS")
    print("="*60)

    print("\n📊 Results by AI Alignment Level:")
    print("-" * 40)

    for align in df['ai_alignment_level'].unique():
        subset = df[df['ai_alignment_level'] == align]
        print(f"\nAlignment = {align}:")
        print(f"  SECI exploit: {subset['final_seci_exploit'].mean():.4f} ± {subset['final_seci_exploit'].std():.4f}")
        print(f"  SECI explore: {subset['final_seci_explore'].mean():.4f} ± {subset['final_seci_explore'].std():.4f}")
        print(f"  MAE exploit:  {subset['final_mae_exploit'].mean():.4f} ± {subset['final_mae_exploit'].std():.4f}")
        print(f"  MAE explore:  {subset['final_mae_explore'].mean():.4f} ± {subset['final_mae_explore'].std():.4f}")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # SECI by alignment
    ax = axes[0, 0]
    for share in df['share_exploitative'].unique():
        subset = df[df['share_exploitative'] == share].groupby('ai_alignment_level').agg({
            'final_seci_exploit': 'mean'
        })
        ax.plot(subset.index, subset['final_seci_exploit'], marker='o', label=f'{int(share*100)}% exploiters')
    ax.set_xlabel('AI Alignment Level')
    ax.set_ylabel('SECI (Exploiters)')
    ax.set_title('Social Echo Chamber Index by Alignment')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # MAE by alignment
    ax = axes[0, 1]
    for share in df['share_exploitative'].unique():
        subset = df[df['share_exploitative'] == share].groupby('ai_alignment_level').agg({
            'final_mae_exploit': 'mean'
        })
        ax.plot(subset.index, subset['final_mae_exploit'], marker='s', label=f'{int(share*100)}% exploiters')
    ax.set_xlabel('AI Alignment Level')
    ax.set_ylabel('MAE (Exploiters)')
    ax.set_title('Belief Error by Alignment')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # SECI by share exploitative
    ax = axes[1, 0]
    for align in df['ai_alignment_level'].unique():
        subset = df[df['ai_alignment_level'] == align].groupby('share_exploitative').agg({
            'final_seci_exploit': 'mean'
        })
        ax.plot(subset.index, subset['final_seci_exploit'], marker='o', label=f'align={align}')
    ax.set_xlabel('Share Exploitative')
    ax.set_ylabel('SECI (Exploiters)')
    ax.set_title('Echo Chamber by Population Composition')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # MAE comparison exploit vs explore
    ax = axes[1, 1]
    exploit_mae = df.groupby('ai_alignment_level')['final_mae_exploit'].mean()
    explore_mae = df.groupby('ai_alignment_level')['final_mae_explore'].mean()
    x = range(len(exploit_mae))
    width = 0.35
    ax.bar([i - width/2 for i in x], exploit_mae.values, width, label='Exploiters', color='coral')
    ax.bar([i + width/2 for i in x], explore_mae.values, width, label='Explorers', color='steelblue')
    ax.set_xticks(x)
    ax.set_xticklabels(exploit_mae.index)
    ax.set_xlabel('AI Alignment Level')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Belief Accuracy: Exploiters vs Explorers')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/analysis_phase6.png', dpi=150)
    print(f"\n📈 Plot saved: {RESULTS_DIR}/analysis_phase6.png")
    plt.close()


def print_summary():
    """Print calibration summary."""
    filepath = f'{RESULTS_DIR}/calibration_summary.md'
    if os.path.exists(filepath):
        print("\n" + "="*60)
        print("CALIBRATION SUMMARY")
        print("="*60)
        with open(filepath) as f:
            print(f.read())


def main():
    print("="*60)
    print("CALIBRATION RESULTS ANALYSIS")
    print("="*60)

    if not os.path.exists(RESULTS_DIR):
        print(f"Error: {RESULTS_DIR}/ directory not found")
        print("Run calibration first: python calibration_experiments.py --quick")
        return

    # List available files
    files = os.listdir(RESULTS_DIR)
    print(f"\nFound files: {files}")

    # Analyze each phase
    analyze_phase2()
    analyze_phase3()
    analyze_phase6()
    print_summary()

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nVisualization files saved to {RESULTS_DIR}/:")
    print("  - analysis_phase2.png (D/δ calibration)")
    print("  - analysis_phase3.png (Memory size)")
    print("  - analysis_phase6.png (Validation)")


if __name__ == '__main__':
    main()
