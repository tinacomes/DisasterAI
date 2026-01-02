"""
Enhanced Analysis Script for DisasterAI Experiment Results
============================================================

This script loads existing experiment results and creates:
1. SECI + AECI-Var overlay plots (chamber transformation analysis)
2. Net chamber strength metric and visualizations
3. Enhanced lifecycle plots showing social→AI bubble transitions

Run this in Google Colab or locally after mounting your results directory.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Configuration - CORRECTED PATHS
try:
    from google.colab import drive
    if not os.path.exists('/content/drive/MyDrive'):
        drive.mount('/content/drive')
    RESULTS_DIR = "/content/drive/MyDrive/DisasterAI_results"
    IN_COLAB = True
    print("✓ Running in Google Colab")
except ImportError:
    RESULTS_DIR = "DisasterAI_results"
    IN_COLAB = False
    print("✓ Running locally")

OUTPUT_DIR = os.path.join(RESULTS_DIR, "enhanced_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"✓ Loading results from: {RESULTS_DIR}")
print(f"✓ Saving enhanced plots to: {OUTPUT_DIR}")

#########################################
# Helper Functions
#########################################

def load_results(experiment_name):
    """Load pickled results from an experiment."""
    pkl_file = os.path.join(RESULTS_DIR, f"results_{experiment_name}.pkl")
    if not os.path.exists(pkl_file):
        print(f"⚠ Warning: {pkl_file} not found")
        return None

    with open(pkl_file, 'rb') as f:
        results = pickle.load(f)
    print(f"✓ Loaded {experiment_name}: {len(results)} parameter values")
    return results

def extract_time_series(results_dict, metric_name, agent_idx=None):
    """
    Extract time series data for a metric across all runs.

    Args:
        results_dict: Results for a single parameter value
        metric_name: Name of metric (e.g., 'seci', 'aeci_variance')
        agent_idx: If metric has agent types, index to extract (1=exploit, 2=explor)

    Returns:
        ticks: Array of tick values
        mean: Mean across runs
        lower: 25th percentile
        upper: 75th percentile
    """
    data = results_dict.get(metric_name)

    if data is None:
        return None, None, None, None

    if not isinstance(data, np.ndarray):
        return None, None, None, None

    try:
        if data.ndim == 3:
            # Shape: (runs, ticks, data)
            if agent_idx is not None:
                # Extract specific agent type
                ticks = data[0, :, 0]
                values = data[:, :, agent_idx]
            else:
                # Extract just the value column
                ticks = data[0, :, 0]
                values = data[:, :, 1]

            mean = np.mean(values, axis=0)
            lower = np.percentile(values, 25, axis=0)
            upper = np.percentile(values, 75, axis=0)

            return ticks, mean, lower, upper

        elif data.ndim == 2:
            # Shape: (runs, ticks)
            ticks = np.arange(data.shape[1])
            mean = np.mean(data, axis=0)
            lower = np.percentile(data, 25, axis=0)
            upper = np.percentile(data, 75, axis=0)

            return ticks, mean, lower, upper
    except Exception as e:
        print(f"  Error extracting {metric_name}: {e}")
        return None, None, None, None

    return None, None, None, None

#########################################
# Analysis 1: SECI + AECI Overlay Plots
#########################################

def plot_chamber_transformation(results_dict, param_values, param_name="AI Alignment",
                                 experiment_name="Experiment_B"):
    """
    Create overlay plots showing SECI and AECI-Var together to reveal
    social→AI chamber transformation.
    """
    print(f"\n=== Creating Chamber Transformation Plots for {experiment_name} ===")

    # Create figure with subplots for each parameter value
    num_params = len(param_values)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Echo Chamber Transformation: Social (SECI) → AI (AECI)\n{param_name} Sensitivity",
                 fontsize=16, fontweight='bold')

    axes = axes.flatten()

    for idx, param_val in enumerate(param_values[:6]):  # Show first 6
        if idx >= len(axes):
            break

        ax = axes[idx]
        res = results_dict.get(param_val, {})

        # Extract SECI (exploitative agents)
        ticks_seci, seci_mean, seci_lower, seci_upper = extract_time_series(res, 'seci', agent_idx=1)

        # Extract AECI-Var
        ticks_aeci, aeci_mean, aeci_lower, aeci_upper = extract_time_series(res, 'aeci_variance', agent_idx=None)

        if ticks_seci is not None:
            # Plot SECI (social echo chamber)
            ax.plot(ticks_seci, seci_mean, label='Social Chamber (SECI)',
                   color='#FF6B6B', linewidth=2.5, alpha=0.9)
            ax.fill_between(ticks_seci, seci_lower, seci_upper,
                           color='#FF6B6B', alpha=0.2)

        if ticks_aeci is not None:
            # Plot AECI-Var (AI echo chamber)
            ax.plot(ticks_aeci, aeci_mean, label='AI Chamber (AECI-Var)',
                   color='#4ECDC4', linewidth=2.5, alpha=0.9)
            ax.fill_between(ticks_aeci, aeci_lower, aeci_upper,
                           color='#4ECDC4', alpha=0.2)

        # Add zero line
        ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

        # Shading for chamber regions
        ax.axhspan(-1, -0.05, alpha=0.05, color='red', label='Echo Chamber Zone')
        ax.axhspan(0.05, 1, alpha=0.05, color='green', label='Diverse Zone')

        # Formatting
        ax.set_title(f'{param_name} = {param_val:.2f}', fontweight='bold')
        ax.set_xlabel('Simulation Tick')
        ax.set_ylabel('Chamber Index')
        ax.set_ylim(-1, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)

    # Hide unused subplots
    for idx in range(len(param_values), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"{experiment_name}_chamber_transformation.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

#########################################
# Analysis 2: Net Chamber Strength
#########################################

def calculate_net_chamber_strength(results_dict, param_values):
    """
    Calculate net chamber strength = |SECI| + |AECI-Var|
    Returns final and peak values for each parameter.
    """
    print(f"\n=== Calculating Net Chamber Strength ===")

    net_strength_data = {}

    for param_val in param_values:
        res = results_dict.get(param_val, {})

        # Extract time series
        ticks_seci, seci_mean, _, _ = extract_time_series(res, 'seci', agent_idx=1)
        ticks_aeci, aeci_mean, _, _ = extract_time_series(res, 'aeci_variance', agent_idx=None)

        if seci_mean is not None and aeci_mean is not None:
            # Align arrays (use minimum length)
            min_len = min(len(seci_mean), len(aeci_mean))

            # Net chamber strength = sum of absolute values
            net_strength = np.abs(seci_mean[:min_len]) + np.abs(aeci_mean[:min_len])

            net_strength_data[param_val] = {
                'time_series': net_strength,
                'ticks': ticks_seci[:min_len] if ticks_seci is not None else np.arange(min_len),
                'final': net_strength[-1],
                'peak': np.max(net_strength),
                'mean': np.mean(net_strength)
            }

            print(f"  {param_val:.2f}: Final={net_strength[-1]:.3f}, Peak={np.max(net_strength):.3f}")

    return net_strength_data

def plot_net_chamber_strength(results_dict, param_values, param_name="AI Alignment",
                               experiment_name="Experiment_B"):
    """Plot net chamber strength over time and summary statistics."""
    print(f"\n=== Plotting Net Chamber Strength ===")

    # Calculate net strength
    net_data = calculate_net_chamber_strength(results_dict, param_values)

    if not net_data:
        print("⚠ No data available for net chamber strength")
        return

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Net Echo Chamber Strength = |SECI| + |AECI-Var|\n{param_name} Sensitivity",
                 fontsize=16, fontweight='bold')

    # Panel 1: Time evolution
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(param_values)))

    for idx, param_val in enumerate(sorted(param_values)):
        data = net_data.get(param_val)
        if data:
            ax.plot(data['ticks'], data['time_series'],
                   label=f'{param_val:.2f}', color=colors[idx], linewidth=2)

    ax.set_xlabel('Simulation Tick', fontweight='bold')
    ax.set_ylabel('Net Chamber Strength', fontweight='bold')
    ax.set_title('Evolution Over Time')
    ax.legend(title=param_name, fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 2: Final values
    ax = axes[1]
    final_values = [net_data[p]['final'] for p in sorted(param_values) if p in net_data]

    ax.bar(range(len(param_values)), final_values,
           color=colors, edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(param_values)))
    ax.set_xticklabels([f'{p:.2f}' for p in sorted(param_values)], rotation=45)
    ax.set_xlabel(param_name, fontweight='bold')
    ax.set_ylabel('Net Chamber Strength', fontweight='bold')
    ax.set_title('Final Values (End of Simulation)')
    ax.grid(True, axis='y', alpha=0.3)

    # Panel 3: Peak values
    ax = axes[2]
    peak_values = [net_data[p]['peak'] for p in sorted(param_values) if p in net_data]

    ax.bar(range(len(param_values)), peak_values,
           color=colors, edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(param_values)))
    ax.set_xticklabels([f'{p:.2f}' for p in sorted(param_values)], rotation=45)
    ax.set_xlabel(param_name, fontweight='bold')
    ax.set_ylabel('Net Chamber Strength', fontweight='bold')
    ax.set_title('Peak Values (Maximum During Simulation)')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"{experiment_name}_net_chamber_strength.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

#########################################
# Analysis 3: Chamber Type Dominance
#########################################

def plot_chamber_dominance(results_dict, param_values, param_name="AI Alignment",
                           experiment_name="Experiment_B"):
    """
    Plot which type of chamber dominates (social vs AI) across parameter values.
    """
    print(f"\n=== Analyzing Chamber Dominance ===")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f"Chamber Type Dominance: Social vs AI\n{param_name} Sensitivity",
                 fontsize=16, fontweight='bold')

    # Panel 1: Time evolution of dominance
    ax = axes[0]
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(param_values)))

    for idx, param_val in enumerate(sorted(param_values)):
        res = results_dict.get(param_val, {})

        # Extract data
        ticks_seci, seci_mean, _, _ = extract_time_series(res, 'seci', agent_idx=1)
        ticks_aeci, aeci_mean, _, _ = extract_time_series(res, 'aeci_variance', agent_idx=None)

        if seci_mean is not None and aeci_mean is not None:
            min_len = min(len(seci_mean), len(aeci_mean))

            # Dominance = |AECI| - |SECI|
            # Positive = AI dominates, Negative = Social dominates
            dominance = np.abs(aeci_mean[:min_len]) - np.abs(seci_mean[:min_len])

            ax.plot(ticks_seci[:min_len], dominance,
                   label=f'{param_val:.2f}', color=colors[idx], linewidth=2)

    ax.axhline(0, color='black', linestyle='--', linewidth=2, alpha=0.7)
    ax.axhspan(-1, 0, alpha=0.1, color='red', label='Social Dominant')
    ax.axhspan(0, 1, alpha=0.1, color='blue', label='AI Dominant')
    ax.set_xlabel('Simulation Tick', fontweight='bold')
    ax.set_ylabel('Chamber Dominance (|AECI| - |SECI|)', fontweight='bold')
    ax.set_title('Evolution: Positive=AI Chamber, Negative=Social Chamber')
    ax.legend(title=param_name, fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)

    # Panel 2: Final dominance values
    ax = axes[1]
    final_dominance = []

    for param_val in sorted(param_values):
        res = results_dict.get(param_val, {})

        # Extract final values
        ticks_seci, seci_mean, _, _ = extract_time_series(res, 'seci', agent_idx=1)
        ticks_aeci, aeci_mean, _, _ = extract_time_series(res, 'aeci_variance', agent_idx=None)

        if seci_mean is not None and aeci_mean is not None:
            final_dom = np.abs(aeci_mean[-1]) - np.abs(seci_mean[-1])
            final_dominance.append(final_dom)
        else:
            final_dominance.append(0)

    colors_dom = ['#FF6B6B' if d < 0 else '#4ECDC4' for d in final_dominance]

    bars = ax.barh(range(len(param_values)), final_dominance,
                   color=colors_dom, edgecolor='black', linewidth=1.5)
    ax.set_yticks(range(len(param_values)))
    ax.set_yticklabels([f'{p:.2f}' for p in sorted(param_values)])
    ax.set_ylabel(param_name, fontweight='bold')
    ax.set_xlabel('Final Dominance (|AECI| - |SECI|)', fontweight='bold')
    ax.set_title('Final State: Red=Social Chamber, Teal=AI Chamber')
    ax.axvline(0, color='black', linestyle='--', linewidth=2)
    ax.grid(True, axis='x', alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, final_dominance)):
        label_x = val + (0.02 if val > 0 else -0.02)
        ax.text(label_x, i, f'{val:.3f}', va='center',
               ha='left' if val > 0 else 'right', fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"{experiment_name}_chamber_dominance.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

#########################################
# Main Analysis
#########################################

def analyze_experiment(experiment_name, param_name, result_key=None):
    """Run all analyses for an experiment."""
    print(f"\n{'='*60}")
    print(f"ANALYZING {experiment_name}")
    print(f"{'='*60}")

    # Load results
    if result_key is None:
        result_key = experiment_name

    results = load_results(result_key)

    if results is None:
        print(f"⚠ Skipping {experiment_name} - no results found")
        return

    param_values = sorted(list(results.keys()))
    print(f"Parameter values: {param_values}")

    # Run analyses
    plot_chamber_transformation(results, param_values, param_name, experiment_name)
    plot_net_chamber_strength(results, param_values, param_name, experiment_name)
    plot_chamber_dominance(results, param_values, param_name, experiment_name)

    print(f"\n✓ Completed analysis for {experiment_name}")

#########################################
# Run All Analyses
#########################################

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ENHANCED ANALYSIS OF DISASTERAI EXPERIMENTS")
    print("="*60)

    # Analyze Experiment A
    analyze_experiment(
        experiment_name="Experiment_A",
        param_name="Share Exploitative",
        result_key="experiment_A"
    )

    # Analyze Experiment B
    analyze_experiment(
        experiment_name="Experiment_B",
        param_name="AI Alignment",
        result_key="AI_Alignment_Tipping_Point"
    )

    print("\n" + "="*60)
    print("✓ ALL ANALYSES COMPLETE")
    print(f"✓ Enhanced plots saved to: {OUTPUT_DIR}")
    print("="*60)
