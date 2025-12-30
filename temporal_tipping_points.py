#!/usr/bin/env python3
"""
Temporal tipping point detection and visualization.

Analyzes WHEN transitions occur during simulations (which tick),
separately for exploit and explor agent types.
"""

import numpy as np
import matplotlib.pyplot as plt


def detect_temporal_tipping_points_single_run(result_data, verbose=False):
    """
    Detect when transitions occur during a single simulation run.

    Returns dictionary mapping transition types to tick numbers (or None if not detected).
    Separate detection for exploit and explor agent types where applicable.

    Args:
        result_data: Dictionary with time series data from one simulation run
        verbose: If True, print diagnostic information

    Returns:
        dict: Transition name → tick number (or None)
    """
    tipping_ticks = {
        'ai_trust_gt_friend_exploit': None,
        'ai_trust_gt_friend_explor': None,
        'seci_crosses_zero_exploit': None,
        'seci_crosses_zero_explor': None,
        'aeci_var_crosses_zero': None,
        'exploit_ai_pref_gt_50': None,
        'explor_ai_pref_gt_50': None,
        'info_diversity_surge': None
    }

    # Extract time series data
    trust = result_data.get("trust_stats", np.array([]))
    seci = result_data.get("seci", np.array([]))
    aeci_var = result_data.get("aeci_variance", np.array([]))
    aeci = result_data.get("aeci", np.array([]))
    info_div = result_data.get("info_diversity", np.array([]))

    # 1. AI trust > Friend trust (separately for exploit and explor)
    if trust.size > 0 and trust.ndim >= 2:
        num_ticks = trust.shape[0]

        # Exploit agents: columns 1 (AI), 2 (Friend)
        if trust.shape[1] > 2:
            for tick in range(1, num_ticks):  # Start from tick 1
                ai_trust_exploit = trust[tick, 1]
                friend_trust_exploit = trust[tick, 2]
                if ai_trust_exploit > friend_trust_exploit and tipping_ticks['ai_trust_gt_friend_exploit'] is None:
                    tipping_ticks['ai_trust_gt_friend_exploit'] = tick
                    break

        # Explor agents: columns 4 (AI), 5 (Friend)
        if trust.shape[1] > 5:
            for tick in range(1, num_ticks):
                ai_trust_explor = trust[tick, 4]
                friend_trust_explor = trust[tick, 5]
                if ai_trust_explor > friend_trust_explor and tipping_ticks['ai_trust_gt_friend_explor'] is None:
                    tipping_ticks['ai_trust_gt_friend_explor'] = tick
                    break

    # 2. SECI crosses zero (separately for exploit and explor)
    if seci.size > 0 and seci.ndim >= 2:
        num_ticks = seci.shape[0]

        # Exploit: column 1
        if seci.shape[1] > 1:
            for tick in range(1, num_ticks):
                if seci[tick-1, 1] < -0.05 and seci[tick, 1] > -0.05:
                    tipping_ticks['seci_crosses_zero_exploit'] = tick
                    break

        # Explor: column 2
        if seci.shape[1] > 2:
            for tick in range(1, num_ticks):
                if seci[tick-1, 2] < -0.05 and seci[tick, 2] > -0.05:
                    tipping_ticks['seci_crosses_zero_explor'] = tick
                    break

    # 3. AECI-Var crosses zero (whole population)
    if aeci_var.size > 0:
        if aeci_var.ndim == 2 and aeci_var.shape[1] >= 2:
            # Format: (ticks, 2) where col 0=tick, col 1=value
            num_ticks = aeci_var.shape[0]
            for tick in range(1, num_ticks):
                if aeci_var[tick-1, 1] < -0.05 and aeci_var[tick, 1] > -0.05:
                    tipping_ticks['aeci_var_crosses_zero'] = int(aeci_var[tick, 0])  # Use tick column
                    break
        elif hasattr(aeci_var, 'dtype') and aeci_var.dtype.names is not None:
            # Structured array
            if 'tick' in aeci_var.dtype.names and 'value' in aeci_var.dtype.names:
                for i in range(1, len(aeci_var)):
                    if aeci_var[i-1]['value'] < -0.05 and aeci_var[i]['value'] > -0.05:
                        tipping_ticks['aeci_var_crosses_zero'] = int(aeci_var[i]['tick'])
                        break

    # 4. AI preference > 50% (separately for exploit and explor)
    if aeci.size > 0 and aeci.ndim >= 2:
        num_ticks = aeci.shape[0]

        # Exploit: column 1
        if aeci.shape[1] > 1:
            for tick in range(num_ticks):
                if aeci[tick, 1] >= 0.5:
                    tipping_ticks['exploit_ai_pref_gt_50'] = tick
                    break

        # Explor: column 2
        if aeci.shape[1] > 2:
            for tick in range(num_ticks):
                if aeci[tick, 2] >= 0.5:
                    tipping_ticks['explor_ai_pref_gt_50'] = tick
                    break

    # 5. Info diversity surge (50% increase from baseline)
    if info_div.size > 0 and info_div.ndim >= 2:
        num_ticks = info_div.shape[0]

        # Use average of exploit and explor
        if info_div.shape[1] > 2:
            # Establish baseline (first 10 ticks)
            baseline_ticks = min(10, num_ticks // 4)
            if baseline_ticks > 0:
                baseline_div = np.mean([info_div[:baseline_ticks, 1], info_div[:baseline_ticks, 2]])

                if baseline_div > 0:
                    for tick in range(baseline_ticks, num_ticks):
                        curr_div = np.mean([info_div[tick, 1], info_div[tick, 2]])
                        if curr_div / baseline_div > 1.5:
                            tipping_ticks['info_diversity_surge'] = tick
                            break

    if verbose:
        print("\nTemporal tipping points detected:")
        for name, tick in tipping_ticks.items():
            if tick is not None:
                print(f"  {name}: tick {tick}")

    return tipping_ticks


def analyze_temporal_tipping_points(results_dict, param_values):
    """
    Analyze temporal tipping points across all parameter values and runs.

    Args:
        results_dict: Dictionary mapping parameter values to aggregated results
        param_values: List of parameter values (e.g., alignment levels)

    Returns:
        dict: Nested dict structure:
            {transition_name: {param_value: [tick_run1, tick_run2, ...]}}
    """
    print("\n=== Analyzing Temporal Tipping Points ===")

    # Initialize storage for all transitions
    temporal_data = {
        'ai_trust_gt_friend_exploit': {},
        'ai_trust_gt_friend_explor': {},
        'seci_crosses_zero_exploit': {},
        'seci_crosses_zero_explor': {},
        'aeci_var_crosses_zero': {},
        'exploit_ai_pref_gt_50': {},
        'explor_ai_pref_gt_50': {},
        'info_diversity_surge': {}
    }

    for param_val in param_values:
        res = results_dict.get(param_val, {})

        print(f"\nAnalyzing {param_val}:")

        # Extract individual run data
        # The aggregated data has shape (runs, ticks, cols)
        trust = res.get("trust_stats", np.array([]))
        seci = res.get("seci", np.array([]))
        aeci_var = res.get("aeci_variance", np.array([]))
        aeci = res.get("aeci", np.array([]))
        info_div = res.get("info_diversity", np.array([]))

        if trust.ndim != 3:
            print(f"  Skipping - trust data has unexpected shape: {trust.shape if hasattr(trust, 'shape') else 'N/A'}")
            continue

        num_runs = trust.shape[0]
        print(f"  Processing {num_runs} runs...")

        # Initialize lists for this parameter value
        for transition_name in temporal_data.keys():
            temporal_data[transition_name][param_val] = []

        # Analyze each run
        for run_idx in range(num_runs):
            # Extract single run data
            run_data = {
                "trust_stats": trust[run_idx, :, :] if trust.ndim == 3 else np.array([]),
                "seci": seci[run_idx, :, :] if seci.ndim == 3 else np.array([]),
                "aeci_variance": aeci_var[run_idx, :, :] if aeci_var.ndim == 3 else aeci_var,
                "aeci": aeci[run_idx, :, :] if aeci.ndim == 3 else np.array([]),
                "info_diversity": info_div[run_idx, :, :] if info_div.ndim == 3 else np.array([])
            }

            # Detect tipping points for this run
            tipping_ticks = detect_temporal_tipping_points_single_run(run_data, verbose=False)

            # Store results
            for transition_name, tick in tipping_ticks.items():
                temporal_data[transition_name][param_val].append(tick)

        # Print summary for this parameter value
        for transition_name in temporal_data.keys():
            ticks = [t for t in temporal_data[transition_name][param_val] if t is not None]
            if ticks:
                mean_tick = np.mean(ticks)
                std_tick = np.std(ticks)
                detection_rate = len(ticks) / num_runs * 100
                print(f"  {transition_name}: tick {mean_tick:.1f}±{std_tick:.1f} ({detection_rate:.0f}% of runs)")

    return temporal_data


def plot_temporal_tipping_points(temporal_data, param_values, param_name="AI Alignment"):
    """
    Visualize how transition timing changes with parameter values.

    Shows separate lines for exploit and explor agent types.
    """
    print("\n=== Generating Temporal Tipping Point Visualization ===")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Transition Timing vs {param_name}\n(When do behavioral shifts occur during simulation?)",
                fontsize=16, fontweight='bold')

    # Panel 1: AI Trust > Friend Trust
    ax = axes[0, 0]
    plot_transition_timing(ax, temporal_data, param_values,
                          ['ai_trust_gt_friend_exploit', 'ai_trust_gt_friend_explor'],
                          ['Exploitative', 'Exploratory'],
                          ['maroon', 'salmon'],
                          "AI Trust Overtakes Friend Trust",
                          param_name)

    # Panel 2: SECI Crosses Zero
    ax = axes[0, 1]
    plot_transition_timing(ax, temporal_data, param_values,
                          ['seci_crosses_zero_exploit', 'seci_crosses_zero_explor'],
                          ['Exploitative', 'Exploratory'],
                          ['darkblue', 'skyblue'],
                          "Social Echo Chamber Breaks (SECI → 0)",
                          param_name)

    # Panel 3: AI Preference > 50%
    ax = axes[1, 0]
    plot_transition_timing(ax, temporal_data, param_values,
                          ['exploit_ai_pref_gt_50', 'explor_ai_pref_gt_50'],
                          ['Exploitative', 'Exploratory'],
                          ['darkgreen', 'lightgreen'],
                          "AI Query Ratio > 50%",
                          param_name)

    # Panel 4: AECI-Var and Info Diversity
    ax = axes[1, 1]
    plot_transition_timing(ax, temporal_data, param_values,
                          ['aeci_var_crosses_zero', 'info_diversity_surge'],
                          ['AECI-Var → 0', 'Info Div Surge'],
                          ['magenta', 'orange'],
                          "System-Wide Transitions",
                          param_name)

    plt.tight_layout()
    plt.savefig("agent_model_results/temporal_tipping_points.png", dpi=300, bbox_inches='tight')
    print("✓ Temporal tipping points plot saved")
    plt.show()


def plot_transition_timing(ax, temporal_data, param_values, transition_names, labels, colors, title, param_name):
    """Helper to plot transition timing for a set of transitions."""

    for transition_name, label, color in zip(transition_names, labels, colors):
        data = temporal_data.get(transition_name, {})

        means = []
        stds = []
        detection_rates = []

        for param_val in param_values:
            ticks = [t for t in data.get(param_val, []) if t is not None]

            if ticks:
                means.append(np.mean(ticks))
                stds.append(np.std(ticks))
                detection_rates.append(len(ticks) / len(data.get(param_val, [1])) * 100)
            else:
                means.append(np.nan)
                stds.append(np.nan)
                detection_rates.append(0)

        # Plot mean with error bars
        valid_indices = ~np.isnan(means)
        if np.any(valid_indices):
            valid_params = np.array(param_values)[valid_indices]
            valid_means = np.array(means)[valid_indices]
            valid_stds = np.array(stds)[valid_indices]

            ax.errorbar(valid_params, valid_means, yerr=valid_stds,
                       marker='o', markersize=8, capsize=5, capthick=2,
                       linewidth=2, alpha=0.8, label=label, color=color)

    ax.set_xlabel(param_name, fontsize=11, fontweight='bold')
    ax.set_ylabel("Tick (when transition occurs)", fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_ylim(bottom=0)

    # Invert y-axis so earlier transitions (lower tick) are at top
    # ax.invert_yaxis()
