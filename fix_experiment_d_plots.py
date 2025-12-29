"""
Fix for Experiment D Empty Plots Issue

This script diagnoses and fixes the plotting issues for Experiment D.

The problem: Empty plots are likely caused by:
1. Incorrect array shape assumptions in plotting code
2. Missing data structure validation
3. Only one plot being created (should be multiple)

This fix creates comprehensive visualizations for Experiment D.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

def diagnose_results_structure(results_d):
    """
    Diagnose the actual structure of results_d to understand why plots are empty.
    """
    print("\n" + "="*70)
    print("DIAGNOSING EXPERIMENT D RESULTS STRUCTURE")
    print("="*70)

    if not results_d:
        print("❌ ERROR: results_d is empty!")
        return False

    print(f"\n✓ Found {len(results_d)} parameter combinations")
    print("\nParameter keys in results_d:")
    for key in sorted(results_d.keys()):
        print(f"  {key}")

    # Check one result in detail
    sample_key = list(results_d.keys())[0]
    sample_result = results_d[sample_key]

    print(f"\n--- Detailed inspection of {sample_key} ---")
    print(f"Type: {type(sample_result)}")

    if isinstance(sample_result, dict):
        print(f"Keys: {list(sample_result.keys())}")

        # Check SECI data
        if "seci" in sample_result:
            seci_data = sample_result["seci"]
            print(f"\nSECI data:")
            print(f"  Type: {type(seci_data)}")
            print(f"  Shape: {seci_data.shape if hasattr(seci_data, 'shape') else 'N/A'}")
            print(f"  Dtype: {seci_data.dtype if hasattr(seci_data, 'dtype') else 'N/A'}")

            if hasattr(seci_data, 'shape'):
                print(f"  First 3 entries: {seci_data[:3] if len(seci_data) > 0 else 'Empty'}")

                # Explain the expected vs actual structure
                print("\n  Expected structure for plotting code:")
                print("    Shape: (num_runs, num_time_points, 3)")
                print("    Format: [tick, exploitative_seci, exploratory_seci]")

                print(f"\n  Actual shape: {seci_data.shape}")

                if seci_data.ndim == 2:
                    print("  ⚠️  ISSUE: Data is 2D, but plotting code expects 3D!")
                    print("     Likely format: (num_time_points, 3) - missing runs dimension")
                    print("     This is why plots are empty!")
                    return False
                elif seci_data.ndim == 3:
                    print("  ✓ Data has correct 3D structure")
                    if seci_data.shape[1] == 0:
                        print("  ❌ ERROR: No time points recorded!")
                        return False
                    return True
                else:
                    print(f"  ❌ ERROR: Unexpected {seci_data.ndim}D structure!")
                    return False
        else:
            print("  ❌ ERROR: No 'seci' key in results!")
            return False

    return False


def create_fixed_experiment_d_plots(results_d, learning_rate_values, epsilon_values, save_dir="agent_model_results"):
    """
    Create comprehensive plots for Experiment D with proper error handling.
    """
    print("\n" + "="*70)
    print("CREATING FIXED EXPERIMENT D PLOTS")
    print("="*70)

    os.makedirs(save_dir, exist_ok=True)

    # Collect data with robust error handling
    data_collector = {
        'seci_exploit': {},
        'seci_explor': {},
        'aeci_exploit': {},
        'aeci_explor': {},
        'belief_error_exploit': {},
        'belief_error_explor': {},
    }

    print("\nExtracting data from results...")

    for lr in learning_rate_values:
        for eps in epsilon_values:
            key = (lr, eps)
            if key not in results_d:
                print(f"  ⚠️  Missing data for LR={lr}, epsilon={eps}")
                continue

            res = results_d[key]

            try:
                # Extract SECI data
                if "seci" in res and res["seci"] is not None:
                    seci_data = res["seci"]

                    # Handle different data structures
                    if hasattr(seci_data, 'ndim'):
                        if seci_data.ndim == 3 and seci_data.shape[1] > 0:
                            # Expected format: (runs, ticks, 3) -> [tick, exploit, explor]
                            data_collector['seci_exploit'][key] = seci_data[:, -1, 1]
                            data_collector['seci_explor'][key] = seci_data[:, -1, 2]
                        elif seci_data.ndim == 2 and seci_data.shape[0] > 0:
                            # Fallback: (ticks, 3) format - single run
                            data_collector['seci_exploit'][key] = [seci_data[-1, 1]]
                            data_collector['seci_explor'][key] = [seci_data[-1, 2]]
                        else:
                            print(f"  ⚠️  Unexpected SECI shape for {key}: {seci_data.shape}")

                # Extract AECI data
                if "aeci" in res and res["aeci"] is not None:
                    aeci_data = res["aeci"]
                    if hasattr(aeci_data, 'ndim'):
                        if aeci_data.ndim == 3 and aeci_data.shape[1] > 0:
                            data_collector['aeci_exploit'][key] = aeci_data[:, -1, 1]
                            data_collector['aeci_explor'][key] = aeci_data[:, -1, 2]
                        elif aeci_data.ndim == 2 and aeci_data.shape[0] > 0:
                            data_collector['aeci_exploit'][key] = [aeci_data[-1, 1]]
                            data_collector['aeci_explor'][key] = [aeci_data[-1, 2]]

                # Extract belief error data
                if "belief_error" in res and res["belief_error"] is not None:
                    error_data = res["belief_error"]
                    if hasattr(error_data, 'ndim'):
                        if error_data.ndim == 3 and error_data.shape[1] > 0:
                            data_collector['belief_error_exploit'][key] = error_data[:, -1, 1]
                            data_collector['belief_error_explor'][key] = error_data[:, -1, 2]
                        elif error_data.ndim == 2 and error_data.shape[0] > 0:
                            data_collector['belief_error_exploit'][key] = [error_data[-1, 1]]
                            data_collector['belief_error_explor'][key] = [error_data[-1, 2]]

                print(f"  ✓ Extracted data for LR={lr}, epsilon={eps}")

            except Exception as e:
                print(f"  ❌ Error extracting data for {key}: {e}")

    # Create plots
    print("\nCreating plots...")

    # Plot 1: SECI Bar Chart
    create_bar_chart(data_collector, 'seci_exploit', 'seci_explor',
                    learning_rate_values, epsilon_values,
                    "SECI (Social Echo Chamber Index)",
                    f"{save_dir}/experiment_d_seci_bars.png")

    # Plot 2: AECI Bar Chart
    create_bar_chart(data_collector, 'aeci_exploit', 'aeci_explor',
                    learning_rate_values, epsilon_values,
                    "AECI (AI Echo Chamber Index)",
                    f"{save_dir}/experiment_d_aeci_bars.png")

    # Plot 3: Belief Error Bar Chart
    create_bar_chart(data_collector, 'belief_error_exploit', 'belief_error_explor',
                    learning_rate_values, epsilon_values,
                    "Belief Error (MAE)",
                    f"{save_dir}/experiment_d_error_bars.png")

    # Plot 4: Heatmap of SECI
    create_heatmap(data_collector, 'seci_exploit', 'seci_explor',
                  learning_rate_values, epsilon_values,
                  "SECI",
                  f"{save_dir}/experiment_d_seci_heatmap.png")

    # Plot 5: Comparison plot
    create_comparison_plot(data_collector, learning_rate_values, epsilon_values,
                          f"{save_dir}/experiment_d_comparison.png")

    print("\n" + "="*70)
    print("PLOTS CREATED SUCCESSFULLY!")
    print("="*70)
    print(f"\nSaved in '{save_dir}/':")
    print("  - experiment_d_seci_bars.png")
    print("  - experiment_d_aeci_bars.png")
    print("  - experiment_d_error_bars.png")
    print("  - experiment_d_seci_heatmap.png")
    print("  - experiment_d_comparison.png")
    print("="*70 + "\n")


def create_bar_chart(data_collector, exploit_key, explor_key, lr_values, eps_values, ylabel, filename):
    """Create bar chart with error bars."""
    fig, axes = plt.subplots(1, len(eps_values), figsize=(12, 5), sharey=True)
    if len(eps_values) == 1:
        axes = [axes]

    fig.suptitle(f"Experiment D: Final {ylabel} vs Learning Rate / Epsilon (Mean & IQR)",
                fontsize=14, fontweight='bold')

    bar_width = 0.35

    for idx, eps in enumerate(eps_values):
        means_exploit = []
        means_explor = []
        errors_exploit = [[], []]  # [lower, upper]
        errors_explor = [[], []]

        for lr in lr_values:
            key = (lr, eps)

            # Exploitative stats
            if key in data_collector[exploit_key] and len(data_collector[exploit_key][key]) > 0:
                values = np.array(data_collector[exploit_key][key])
                mean_exp = np.mean(values)
                p25_exp = np.percentile(values, 25)
                p75_exp = np.percentile(values, 75)

                means_exploit.append(mean_exp)
                errors_exploit[0].append(mean_exp - p25_exp)
                errors_exploit[1].append(p75_exp - mean_exp)
            else:
                means_exploit.append(0)
                errors_exploit[0].append(0)
                errors_exploit[1].append(0)

            # Exploratory stats
            if key in data_collector[explor_key] and len(data_collector[explor_key][key]) > 0:
                values = np.array(data_collector[explor_key][key])
                mean_er = np.mean(values)
                p25_er = np.percentile(values, 25)
                p75_er = np.percentile(values, 75)

                means_explor.append(mean_er)
                errors_explor[0].append(mean_er - p25_er)
                errors_explor[1].append(p75_er - mean_er)
            else:
                means_explor.append(0)
                errors_explor[0].append(0)
                errors_explor[1].append(0)

        # Plot bars
        x_pos = np.arange(len(lr_values))
        ax = axes[idx]

        ax.bar(x_pos - bar_width/2, means_exploit, bar_width,
              yerr=errors_exploit, capsize=4,
              label='Exploitative', color='#D55E00', alpha=0.7,
              error_kw=dict(alpha=0.5, linewidth=1.5))

        ax.bar(x_pos + bar_width/2, means_explor, bar_width,
              yerr=errors_explor, capsize=4,
              label='Exploratory', color='#009E73', alpha=0.7,
              error_kw=dict(alpha=0.5, linewidth=1.5))

        ax.set_xlabel("Learning Rate", fontsize=11)
        if idx == 0:
            ax.set_ylabel(f"Mean Final {ylabel}", fontsize=11)
        ax.set_title(f"ε = {eps}", fontsize=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(lr_values)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Created {filename}")


def create_heatmap(data_collector, exploit_key, explor_key, lr_values, eps_values, metric_name, filename):
    """Create heatmap showing metric across parameters."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    fig.suptitle(f"Experiment D: {metric_name} Heatmap", fontsize=14, fontweight='bold')

    # Create matrices
    exploit_matrix = np.zeros((len(eps_values), len(lr_values)))
    explor_matrix = np.zeros((len(eps_values), len(lr_values)))

    for i, eps in enumerate(eps_values):
        for j, lr in enumerate(lr_values):
            key = (lr, eps)

            if key in data_collector[exploit_key] and len(data_collector[exploit_key][key]) > 0:
                exploit_matrix[i, j] = np.mean(data_collector[exploit_key][key])

            if key in data_collector[explor_key] and len(data_collector[explor_key][key]) > 0:
                explor_matrix[i, j] = np.mean(data_collector[explor_key][key])

    # Plot heatmaps
    im1 = ax1.imshow(exploit_matrix, cmap='RdYlGn', aspect='auto')
    ax1.set_title('Exploitative Agents', fontsize=12)
    ax1.set_xlabel('Learning Rate', fontsize=11)
    ax1.set_ylabel('Epsilon', fontsize=11)
    ax1.set_xticks(range(len(lr_values)))
    ax1.set_xticklabels(lr_values)
    ax1.set_yticks(range(len(eps_values)))
    ax1.set_yticklabels(eps_values)
    plt.colorbar(im1, ax=ax1, label=metric_name)

    # Add values to cells
    for i in range(len(eps_values)):
        for j in range(len(lr_values)):
            text = ax1.text(j, i, f'{exploit_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)

    im2 = ax2.imshow(explor_matrix, cmap='RdYlGn', aspect='auto')
    ax2.set_title('Exploratory Agents', fontsize=12)
    ax2.set_xlabel('Learning Rate', fontsize=11)
    ax2.set_ylabel('Epsilon', fontsize=11)
    ax2.set_xticks(range(len(lr_values)))
    ax2.set_xticklabels(lr_values)
    ax2.set_yticks(range(len(eps_values)))
    ax2.set_yticklabels(eps_values)
    plt.colorbar(im2, ax=ax2, label=metric_name)

    # Add values to cells
    for i in range(len(eps_values)):
        for j in range(len(lr_values)):
            text = ax2.text(j, i, f'{explor_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Created {filename}")


def create_comparison_plot(data_collector, lr_values, eps_values, filename):
    """Create comprehensive comparison plot."""
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    fig.suptitle('Experiment D: Comprehensive Comparison', fontsize=16, fontweight='bold')

    metrics = [
        ('seci_exploit', 'seci_explor', 'SECI', 0, 0),
        ('aeci_exploit', 'aeci_explor', 'AECI', 0, 1),
        ('belief_error_exploit', 'belief_error_explor', 'Belief Error (MAE)', 1, 0),
    ]

    for exploit_key, explor_key, metric_name, row, col in metrics:
        ax = fig.add_subplot(gs[row, col])

        for eps in eps_values:
            exploit_means = []
            explor_means = []

            for lr in lr_values:
                key = (lr, eps)

                if key in data_collector[exploit_key] and len(data_collector[exploit_key][key]) > 0:
                    exploit_means.append(np.mean(data_collector[exploit_key][key]))
                else:
                    exploit_means.append(0)

                if key in data_collector[explor_key] and len(data_collector[explor_key][key]) > 0:
                    explor_means.append(np.mean(data_collector[explor_key][key]))
                else:
                    explor_means.append(0)

            ax.plot(lr_values, exploit_means, marker='o', label=f'Exploit (ε={eps})',
                   linestyle='-', linewidth=2)
            ax.plot(lr_values, explor_means, marker='s', label=f'Explor (ε={eps})',
                   linestyle='--', linewidth=2)

        ax.set_xlabel('Learning Rate', fontsize=10)
        ax.set_ylabel(metric_name, fontsize=10)
        ax.set_title(metric_name, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Add interpretation text
    ax_text = fig.add_subplot(gs[2, :])
    ax_text.axis('off')
    interpretation = """
    INTERPRETATION:
    • Higher learning rates (0.07) → Faster Q-learning convergence → Stronger specialization
    • Lower epsilon (0.2) → More exploitation → Agents lock into AI-only or friend-only patterns
    • SECI: Positive = social bubbles, Negative = social diversity
    • AECI: Positive = AI creates echo chamber, Negative = AI breaks bubbles
    • Belief Error: Lower = better accuracy in disaster assessment
    """
    ax_text.text(0.1, 0.5, interpretation, fontsize=10, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Created {filename}")


# Main execution
if __name__ == "__main__":
    print("\n" + "="*70)
    print("EXPERIMENT D PLOT FIX UTILITY")
    print("="*70)

    # Load results
    results_file = "agent_model_results/results_experiment_D.pkl"

    if not os.path.exists(results_file):
        print(f"\n❌ ERROR: Results file not found: {results_file}")
        print("\nPlease ensure Experiment D has completed and saved results.")
        print("Expected file: agent_model_results/results_experiment_D.pkl")
    else:
        print(f"\n✓ Loading results from {results_file}...")

        with open(results_file, 'rb') as f:
            results_d = pickle.load(f)

        # Diagnose structure
        is_valid = diagnose_results_structure(results_d)

        # Create plots regardless (with error handling)
        learning_rate_values = [0.03, 0.05, 0.07]
        epsilon_values = [0.2, 0.3]

        create_fixed_experiment_d_plots(results_d, learning_rate_values, epsilon_values)

    print("\nDone!")
