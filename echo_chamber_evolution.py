#!/usr/bin/env python3
"""
Echo Chamber Evolution Analysis

Tracks the RISE and FALL of echo chambers throughout simulations.
Shows formation, peak, and dissolution phases.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def extract_echo_chamber_lifecycle(results_dict, param_values):
    """
    Extract peak metrics and full time series for echo chamber evolution.

    Returns:
        dict with keys:
            - 'seci_time_series': {param_val: (mean_exploit, std_exploit, mean_explor, std_explor, ticks)}
            - 'aeci_time_series': {param_val: (mean_exploit, std_exploit, mean_explor, std_explor, ticks)}
            - 'aeci_var_time_series': {param_val: (mean, std, ticks)}
            - 'peak_metrics': {param_val: {...}}
    """
    print("\n=== Extracting Echo Chamber Lifecycle Data ===")

    lifecycle_data = {
        'seci_time_series': {},
        'aeci_time_series': {},
        'aeci_var_time_series': {},
        'peak_metrics': {}
    }

    for param_val in param_values:
        res = results_dict.get(param_val, {})

        print(f"\nProcessing {param_val}:")

        # Extract time series data
        seci = res.get("seci", np.array([]))
        aeci = res.get("aeci", np.array([]))
        aeci_var = res.get("aeci_variance", np.array([]))

        if seci.size == 0 or seci.ndim != 3:
            print(f"  Skipping - invalid SECI shape: {seci.shape if hasattr(seci, 'shape') else 'N/A'}")
            continue

        num_runs = seci.shape[0]
        num_ticks = seci.shape[1]

        print(f"  {num_runs} runs, {num_ticks} ticks")

        # SECI time series (exploit and explor separately)
        seci_exploit_all = seci[:, :, 1]  # (runs, ticks)
        seci_explor_all = seci[:, :, 2]

        seci_exploit_mean = np.mean(seci_exploit_all, axis=0)
        seci_exploit_std = np.std(seci_exploit_all, axis=0)
        seci_explor_mean = np.mean(seci_explor_all, axis=0)
        seci_explor_std = np.std(seci_explor_all, axis=0)

        lifecycle_data['seci_time_series'][param_val] = (
            seci_exploit_mean, seci_exploit_std,
            seci_explor_mean, seci_explor_std,
            np.arange(num_ticks)
        )

        # AECI time series
        if aeci.ndim == 3 and aeci.shape[1] > 0:
            aeci_exploit_all = aeci[:, :, 1]
            aeci_explor_all = aeci[:, :, 2]

            aeci_exploit_mean = np.mean(aeci_exploit_all, axis=0)
            aeci_exploit_std = np.std(aeci_exploit_all, axis=0)
            aeci_explor_mean = np.mean(aeci_explor_all, axis=0)
            aeci_explor_std = np.std(aeci_explor_all, axis=0)

            lifecycle_data['aeci_time_series'][param_val] = (
                aeci_exploit_mean, aeci_exploit_std,
                aeci_explor_mean, aeci_explor_std,
                np.arange(num_ticks)
            )

        # AECI-Var time series
        if aeci_var.ndim == 3 and aeci_var.shape[1] > 0:
            aeci_var_all = aeci_var[:, :, 1]  # Value column

            aeci_var_mean = np.mean(aeci_var_all, axis=0)
            aeci_var_std = np.std(aeci_var_all, axis=0)

            lifecycle_data['aeci_var_time_series'][param_val] = (
                aeci_var_mean, aeci_var_std,
                np.arange(num_ticks)
            )

        # Calculate peak metrics
        peak_metrics = {}

        # SECI peaks (most negative = strongest echo chamber)
        seci_exploit_peak_idx = np.argmin(seci_exploit_mean)
        seci_explor_peak_idx = np.argmin(seci_explor_mean)

        peak_metrics['seci_exploit_peak'] = seci_exploit_mean[seci_exploit_peak_idx]
        peak_metrics['seci_exploit_peak_tick'] = seci_exploit_peak_idx
        peak_metrics['seci_explor_peak'] = seci_explor_mean[seci_explor_peak_idx]
        peak_metrics['seci_explor_peak_tick'] = seci_explor_peak_idx

        # AECI peaks (highest ratio)
        if aeci.ndim == 3:
            aeci_exploit_peak_idx = np.argmax(aeci_exploit_mean)
            aeci_explor_peak_idx = np.argmax(aeci_explor_mean)

            peak_metrics['aeci_exploit_peak'] = aeci_exploit_mean[aeci_exploit_peak_idx]
            peak_metrics['aeci_exploit_peak_tick'] = aeci_exploit_peak_idx
            peak_metrics['aeci_explor_peak'] = aeci_explor_mean[aeci_explor_peak_idx]
            peak_metrics['aeci_explor_peak_tick'] = aeci_explor_peak_idx

        # AECI-Var peaks (most negative = strongest belief convergence)
        if aeci_var.ndim == 3:
            # Peak is maximum absolute value
            aeci_var_abs = np.abs(aeci_var_mean)
            aeci_var_peak_idx = np.argmax(aeci_var_abs)

            peak_metrics['aeci_var_peak'] = aeci_var_mean[aeci_var_peak_idx]
            peak_metrics['aeci_var_peak_tick'] = aeci_var_peak_idx

        # Chamber metrics: Use data-driven approach instead of arbitrary threshold
        # Calculate "significant chamber" as ticks where SECI is in bottom 25% of its range
        seci_exploit_p25 = np.percentile(seci_exploit_mean, 25)
        seci_explor_p25 = np.percentile(seci_explor_mean, 25)

        seci_exploit_chamber_ticks = np.sum(seci_exploit_mean < seci_exploit_p25)
        seci_explor_chamber_ticks = np.sum(seci_explor_mean < seci_explor_p25)

        peak_metrics['seci_exploit_duration'] = seci_exploit_chamber_ticks
        peak_metrics['seci_explor_duration'] = seci_explor_chamber_ticks
        peak_metrics['seci_exploit_threshold'] = seci_exploit_p25
        peak_metrics['seci_explor_threshold'] = seci_explor_p25

        lifecycle_data['peak_metrics'][param_val] = peak_metrics

        print(f"  SECI exploit peak: {peak_metrics['seci_exploit_peak']:.3f} at tick {peak_metrics['seci_exploit_peak_tick']}")
        print(f"  SECI explor peak: {peak_metrics['seci_explor_peak']:.3f} at tick {peak_metrics['seci_explor_peak_tick']}")
        if 'aeci_var_peak' in peak_metrics:
            print(f"  AECI-Var peak: {peak_metrics['aeci_var_peak']:.3f} at tick {peak_metrics['aeci_var_peak_tick']}")
        print(f"  Chamber duration (SECI < p25={seci_exploit_p25:.3f}): exploit={seci_exploit_chamber_ticks} ticks, explor={seci_explor_chamber_ticks} ticks")

    return lifecycle_data


def plot_echo_chamber_evolution(lifecycle_data, param_values, param_name="AI Alignment"):
    """
    Comprehensive visualization of echo chamber rise and fall.

    Shows:
    1. Full time series of SECI evolution
    2. Peak strength comparison across alignments
    3. Timing of formation and dissolution
    """
    print("\n=== Generating Echo Chamber Evolution Visualization ===")

    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    fig.suptitle(f"Echo Chamber Lifecycle: Rise and Fall\n(How filter bubbles form, peak, and dissolve)",
                 fontsize=18, fontweight='bold')

    # Color palette for alignment values
    colors = plt.cm.viridis(np.linspace(0.2, 0.95, len(param_values)))

    # ========================================
    # Panel 1: SECI Evolution (Exploitative)
    # ========================================
    ax1 = fig.add_subplot(gs[0, :2])

    for i, param_val in enumerate(param_values):
        if param_val not in lifecycle_data['seci_time_series']:
            continue

        mean_exploit, std_exploit, mean_explor, std_explor, ticks = lifecycle_data['seci_time_series'][param_val]

        # Plot exploitative agents
        ax1.plot(ticks, mean_exploit, color=colors[i], linewidth=2.5,
                label=f"{param_name}={param_val}", alpha=0.9)
        ax1.fill_between(ticks, mean_exploit - std_exploit, mean_exploit + std_exploit,
                         color=colors[i], alpha=0.2)

    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Neutral (SECI=0)')

    ax1.set_xlabel("Simulation Tick", fontsize=12, fontweight='bold')
    ax1.set_ylabel("SECI (Social Echo Chamber Index)", fontsize=12, fontweight='bold')
    ax1.set_title("Exploitative Agents: Echo Chamber Formation & Dissolution", fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.set_ylim([-0.5, 0.2])

    # Add annotations
    ax1.text(0.02, 0.98, "📉 Negative SECI = Friend network homophily\n"
             "(friends have lower belief variance than population)\n\n"
             "⚠️  Limited alignment sensitivity:\n"
             "Social network is fixed at initialization",
             transform=ax1.transAxes, fontsize=8, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # ========================================
    # Panel 2: SECI Evolution (Exploratory)
    # ========================================
    ax2 = fig.add_subplot(gs[1, :2])

    for i, param_val in enumerate(param_values):
        if param_val not in lifecycle_data['seci_time_series']:
            continue

        mean_exploit, std_exploit, mean_explor, std_explor, ticks = lifecycle_data['seci_time_series'][param_val]

        # Plot exploratory agents
        ax2.plot(ticks, mean_explor, color=colors[i], linewidth=2.5,
                label=f"{param_name}={param_val}", alpha=0.9)
        ax2.fill_between(ticks, mean_explor - std_explor, mean_explor + std_explor,
                         color=colors[i], alpha=0.2)

    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Neutral (SECI=0)')

    ax2.set_xlabel("Simulation Tick", fontsize=12, fontweight='bold')
    ax2.set_ylabel("SECI (Social Echo Chamber Index)", fontsize=12, fontweight='bold')
    ax2.set_title("Exploratory Agents: Echo Chamber Formation & Dissolution", fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle=':')
    ax2.set_ylim([-0.5, 0.2])

    # ========================================
    # Panel 3: Peak Strength Comparison
    # ========================================
    ax3 = fig.add_subplot(gs[0, 2])

    peak_seci_exploit = []
    peak_seci_explor = []

    for param_val in param_values:
        if param_val in lifecycle_data['peak_metrics']:
            peak_seci_exploit.append(lifecycle_data['peak_metrics'][param_val]['seci_exploit_peak'])
            peak_seci_explor.append(lifecycle_data['peak_metrics'][param_val]['seci_explor_peak'])
        else:
            peak_seci_exploit.append(0)
            peak_seci_explor.append(0)

    x = np.arange(len(param_values))
    width = 0.35

    ax3.bar(x - width/2, np.abs(peak_seci_exploit), width, label='Exploitative',
            color='maroon', alpha=0.8, edgecolor='black')
    ax3.bar(x + width/2, np.abs(peak_seci_explor), width, label='Exploratory',
            color='salmon', alpha=0.8, edgecolor='black')

    ax3.set_xlabel(param_name, fontsize=11, fontweight='bold')
    ax3.set_ylabel("Peak Echo Chamber Strength\n|SECI|", fontsize=11, fontweight='bold')
    ax3.set_title("Maximum Chamber Strength", fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(param_values)
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y', linestyle=':')

    # ========================================
    # Panel 4: Time to Peak
    # ========================================
    ax4 = fig.add_subplot(gs[1, 2])

    peak_tick_exploit = []
    peak_tick_explor = []

    for param_val in param_values:
        if param_val in lifecycle_data['peak_metrics']:
            peak_tick_exploit.append(lifecycle_data['peak_metrics'][param_val]['seci_exploit_peak_tick'])
            peak_tick_explor.append(lifecycle_data['peak_metrics'][param_val]['seci_explor_peak_tick'])
        else:
            peak_tick_exploit.append(0)
            peak_tick_explor.append(0)

    ax4.bar(x - width/2, peak_tick_exploit, width, label='Exploitative',
            color='darkblue', alpha=0.8, edgecolor='black')
    ax4.bar(x + width/2, peak_tick_explor, width, label='Exploratory',
            color='skyblue', alpha=0.8, edgecolor='black')

    ax4.set_xlabel(param_name, fontsize=11, fontweight='bold')
    ax4.set_ylabel("Time to Peak (ticks)", fontsize=11, fontweight='bold')
    ax4.set_title("When Do Chambers Peak?", fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(param_values)
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y', linestyle=':')

    # ========================================
    # Panel 5: AECI-Var Evolution
    # ========================================
    ax5 = fig.add_subplot(gs[2, :2])

    for i, param_val in enumerate(param_values):
        if param_val not in lifecycle_data['aeci_var_time_series']:
            continue

        mean, std, ticks = lifecycle_data['aeci_var_time_series'][param_val]

        ax5.plot(ticks, mean, color=colors[i], linewidth=2.5,
                label=f"{param_name}={param_val}", alpha=0.9)
        ax5.fill_between(ticks, mean - std, mean + std,
                         color=colors[i], alpha=0.2)

    ax5.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Neutral')

    ax5.set_xlabel("Simulation Tick", fontsize=12, fontweight='bold')
    ax5.set_ylabel("AECI-Var (AI Echo Chamber Index)", fontsize=12, fontweight='bold')
    ax5.set_title("AI Belief Variance Reduction Over Time", fontsize=13, fontweight='bold')
    ax5.legend(loc='best', fontsize=9)
    ax5.grid(True, alpha=0.3, linestyle=':')

    ax5.text(0.02, 0.98, "📉 Negative = AI-reliant agents have\nlower belief variance (convergence)",
             transform=ax5.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # ========================================
    # Panel 6: Chamber Duration
    # ========================================
    ax6 = fig.add_subplot(gs[2, 2])

    duration_exploit = []
    duration_explor = []

    for param_val in param_values:
        if param_val in lifecycle_data['peak_metrics']:
            duration_exploit.append(lifecycle_data['peak_metrics'][param_val]['seci_exploit_duration'])
            duration_explor.append(lifecycle_data['peak_metrics'][param_val]['seci_explor_duration'])
        else:
            duration_exploit.append(0)
            duration_explor.append(0)

    ax6.bar(x - width/2, duration_exploit, width, label='Exploitative',
            color='darkgreen', alpha=0.8, edgecolor='black')
    ax6.bar(x + width/2, duration_explor, width, label='Exploratory',
            color='lightgreen', alpha=0.8, edgecolor='black')

    ax6.set_xlabel(param_name, fontsize=11, fontweight='bold')
    ax6.set_ylabel("Duration (ticks in bottom quartile)", fontsize=11, fontweight='bold')
    ax6.set_title("How Long Do Chambers Persist?\n(Data-driven threshold: 25th percentile)", fontsize=11, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(param_values)
    ax6.legend(loc='best', fontsize=9)
    ax6.grid(True, alpha=0.3, axis='y', linestyle=':')

    plt.savefig("agent_model_results/echo_chamber_evolution.png", dpi=300, bbox_inches='tight')
    print("✓ Echo chamber evolution plot saved")
    plt.show()


def plot_aeci_evolution(lifecycle_data, param_values, param_name="AI Alignment"):
    """
    Visualize AI preference evolution over time.
    Shows how agents shift from friends to AI queries.
    """
    print("\n=== Generating AECI Evolution Visualization ===")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"AI Query Preference Evolution\n(Agent shift from friends to AI)",
                 fontsize=16, fontweight='bold')

    colors = plt.cm.plasma(np.linspace(0.2, 0.95, len(param_values)))

    # Panel 1: Exploitative agents
    for i, param_val in enumerate(param_values):
        if param_val not in lifecycle_data['aeci_time_series']:
            continue

        mean_exploit, std_exploit, mean_explor, std_explor, ticks = lifecycle_data['aeci_time_series'][param_val]

        ax1.plot(ticks, mean_exploit, color=colors[i], linewidth=2.5,
                label=f"{param_name}={param_val}", alpha=0.9)
        ax1.fill_between(ticks, mean_exploit - std_exploit, mean_exploit + std_exploit,
                         color=colors[i], alpha=0.2)

    ax1.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='50% threshold')
    ax1.set_xlabel("Simulation Tick", fontsize=12, fontweight='bold')
    ax1.set_ylabel("AECI (AI Query Ratio)", fontsize=12, fontweight='bold')
    ax1.set_title("Exploitative Agents", fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.set_ylim([0, 1])

    # Panel 2: Exploratory agents
    for i, param_val in enumerate(param_values):
        if param_val not in lifecycle_data['aeci_time_series']:
            continue

        mean_exploit, std_exploit, mean_explor, std_explor, ticks = lifecycle_data['aeci_time_series'][param_val]

        ax2.plot(ticks, mean_explor, color=colors[i], linewidth=2.5,
                label=f"{param_name}={param_val}", alpha=0.9)
        ax2.fill_between(ticks, mean_explor - std_explor, mean_explor + std_explor,
                         color=colors[i], alpha=0.2)

    ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='50% threshold')
    ax2.set_xlabel("Simulation Tick", fontsize=12, fontweight='bold')
    ax2.set_ylabel("AECI (AI Query Ratio)", fontsize=12, fontweight='bold')
    ax2.set_title("Exploratory Agents", fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle=':')
    ax2.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig("agent_model_results/aeci_evolution.png", dpi=300, bbox_inches='tight')
    print("✓ AECI evolution plot saved")
    plt.show()
