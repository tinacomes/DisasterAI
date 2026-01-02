"""
Essential Experiments for Paper Finalization
=============================================

Runs Experiments A, B, and D in one batch for paper completion.

Experiment A: Share of exploitative agents (RQ4)
Experiment B: AI alignment levels (RQ1, RQ2, RQ3)
Experiment D: Q-learning parameter sensitivity (robustness validation)

Estimated runtime: ~3 hours
"""

# NOTE: Copy everything from line 2 onwards from DisasterAI_Model.py
# EXCEPT replace the main section with this:

if __name__ == "__main__":
    import time
    start_time = time.time()

    # Set save directory (Drive if in Colab, local otherwise)
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        save_dir = "/content/drive/MyDrive/DisasterAI_results"
        IN_COLAB = True
        print("✓ Running in Google Colab - results will be saved to Drive")
    except:
        save_dir = "DisasterAI_results"
        IN_COLAB = False
        print("✓ Running locally")

    os.makedirs(save_dir, exist_ok=True)
    print(f"✓ Results directory: {save_dir}")

    # Base parameters
    base_params = {
        "share_exploitative": 0.5,
        "share_of_disaster": 0.2,
        "initial_trust": 0.5,
        "initial_ai_trust": 0.5,
        "number_of_humans": 30,
        "share_confirming": 0.5,
        "disaster_dynamics": 2,
        "shock_probability": 0.1,
        "shock_magnitude": 2,
        "trust_update_mode": "average",
        "ai_alignment_level": 0.0,
        "exploitative_correction_factor": 1.0,
        "width": 30,
        "height": 30,
        "num_ai": 3,
        "max_ticks": 150,
        "debug_mode": False
    }

    num_runs = 10  # 10 runs per parameter value

    print("\n" + "="*70)
    print("ESSENTIAL EXPERIMENTS FOR PAPER FINALIZATION")
    print("="*70)
    print(f"Base parameters: {num_runs} runs per condition")
    print(f"Estimated total runtime: ~3 hours")
    print("="*70)

    ##############################################
    # Experiment A: Share of Exploitative Agents
    ##############################################
    print("\n" + "="*70)
    print("EXPERIMENT A: Share of Exploitative Agents")
    print("="*70)
    print("Tests: How does agent composition affect outcomes? (RQ4)")
    print("Parameter values: [0.2, 0.5, 0.8]")
    print("="*70)

    exp_a_start = time.time()
    share_values = [0.2, 0.5, 0.8]
    file_a_pkl = os.path.join(save_dir, "results_experiment_A.pkl")
    file_a_csv = os.path.join(save_dir, "results_experiment_A.csv")

    print(f"\n▶ Running Experiment A with {len(share_values)} parameter values...")
    results_a = experiment_share_exploitative(base_params, share_values, num_runs)

    # Save results
    with open(file_a_pkl, "wb") as f:
        pickle.dump(results_a, f)
    export_results_to_csv(results_a, share_values, file_a_csv, "Experiment A")

    exp_a_time = (time.time() - exp_a_start) / 60
    print(f"\n✓ Experiment A completed in {exp_a_time:.1f} minutes")
    print(f"✓ Saved to: {file_a_pkl}")

    # Generate plots
    print("\n▶ Generating Experiment A visualizations...")
    for share in share_values:
        results_dict = results_a.get(share, {})
        if results_dict:
            title_suffix = f"(Share Exploitative={share})"
            plot_simulation_overview(results_dict, title_suffix)
            plot_echo_chamber_indices(results_dict, title_suffix)
            plot_trust_evolution(results_dict["trust_stats"], title_suffix)

    # Summary plots
    plot_correct_token_shares_bars(results_a, share_values)
    plot_phase_diagram_bubbles(results_a, share_values, param_name="Share Exploitative")
    plot_tipping_point_waterfall(results_a, share_values, param_name="Share Exploitative")

    ##############################################
    # Experiment B: AI Alignment Levels
    ##############################################
    print("\n" + "="*70)
    print("EXPERIMENT B: AI Alignment Levels")
    print("="*70)
    print("Tests: Do AI agents break or create filter bubbles? (RQ1, RQ2, RQ3)")
    print("Parameter values: [0.0, 0.25, 0.5, 0.75, 1.0]")
    print("="*70)

    exp_b_start = time.time()
    alignment_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    file_b_pkl = os.path.join(save_dir, "results_experiment_B.pkl")

    print(f"\n▶ Running Experiment B with {len(alignment_values)} parameter values...")
    results_b = experiment_alignment_tipping_point(base_params, alignment_values, num_runs=10)

    # Save results
    with open(file_b_pkl, "wb") as f:
        pickle.dump(results_b, f)

    exp_b_time = (time.time() - exp_b_start) / 60
    print(f"\n✓ Experiment B completed in {exp_b_time:.1f} minutes")
    print(f"✓ Saved to: {file_b_pkl}")

    # Generate plots
    print("\n▶ Generating Experiment B visualizations...")
    all_alignment_values = sorted(list(results_b.keys()))

    plot_final_echo_indices_vs_alignment(results_b, all_alignment_values, title_suffix="Tipping Points")
    plot_average_performance_vs_alignment(results_b, all_alignment_values, title_suffix="Tipping Points")
    plot_summary_echo_indices_vs_alignment(results_b, all_alignment_values, title_suffix="Tipping Points")
    plot_summary_performance_vs_alignment(results_b, all_alignment_values, title_suffix="Tipping Points")
    plot_phase_diagram_bubbles(results_b, all_alignment_values, param_name="AI Alignment")
    plot_tipping_point_waterfall(results_b, all_alignment_values, param_name="AI Alignment")

    ##############################################
    # Experiment D: Q-Learning Parameter Sensitivity
    ##############################################
    print("\n" + "="*70)
    print("EXPERIMENT D: Q-Learning Parameter Sensitivity")
    print("="*70)
    print("Tests: Are findings robust to learning parameters? (Validation)")
    print("Parameter values: LR=[0.05, 0.1, 0.15] × Epsilon=[0.2, 0.3, 0.4]")
    print("="*70)

    exp_d_start = time.time()
    learning_rate_values = [0.05, 0.1, 0.15]
    epsilon_values = [0.2, 0.3, 0.4]
    file_d_pkl = os.path.join(save_dir, "results_experiment_D.pkl")

    print(f"\n▶ Running Experiment D with {len(learning_rate_values)}×{len(epsilon_values)} combinations...")
    results_d = experiment_learning_trust(base_params, learning_rate_values, epsilon_values, num_runs)

    # Save results
    with open(file_d_pkl, "wb") as f:
        pickle.dump(results_d, f)

    exp_d_time = (time.time() - exp_d_start) / 60
    print(f"\n✓ Experiment D completed in {exp_d_time:.1f} minutes")
    print(f"✓ Saved to: {file_d_pkl}")

    # Generate plots for Experiment D
    print("\n▶ Generating Experiment D visualizations...")

    # Create heatmaps showing robustness
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Experiment D: Q-Learning Parameter Sensitivity Analysis",
                 fontsize=16, fontweight='bold')

    # Extract final SECI values for exploitative agents
    seci_matrix = np.zeros((len(learning_rate_values), len(epsilon_values)))
    aeci_matrix = np.zeros((len(learning_rate_values), len(epsilon_values)))

    for i, lr in enumerate(learning_rate_values):
        for j, eps in enumerate(epsilon_values):
            res = results_d.get((lr, eps), {})
            seci_data = res.get("seci", np.array([]))
            aeci_data = res.get("aeci_variance", np.array([]))

            if seci_data.size > 0:
                seci_matrix[i, j] = np.mean(seci_data[:, -1, 1])  # Final exploitative SECI
            if aeci_data.size > 0:
                aeci_matrix[i, j] = np.mean(aeci_data[:, -1, 1])  # Final AECI

    # Plot SECI heatmap
    ax = axes[0, 0]
    im1 = ax.imshow(seci_matrix, cmap='RdYlGn_r', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(len(epsilon_values)))
    ax.set_yticks(range(len(learning_rate_values)))
    ax.set_xticklabels([f'{e:.2f}' for e in epsilon_values])
    ax.set_yticklabels([f'{lr:.2f}' for lr in learning_rate_values])
    ax.set_xlabel('Epsilon (Exploration Rate)')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Final SECI (Social Echo Chamber)')
    plt.colorbar(im1, ax=ax)

    # Add values to cells
    for i in range(len(learning_rate_values)):
        for j in range(len(epsilon_values)):
            ax.text(j, i, f'{seci_matrix[i, j]:.2f}',
                   ha='center', va='center', color='black', fontweight='bold')

    # Plot AECI heatmap
    ax = axes[0, 1]
    im2 = ax.imshow(aeci_matrix, cmap='RdYlGn_r', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(len(epsilon_values)))
    ax.set_yticks(range(len(learning_rate_values)))
    ax.set_xticklabels([f'{e:.2f}' for e in epsilon_values])
    ax.set_yticklabels([f'{lr:.2f}' for lr in learning_rate_values])
    ax.set_xlabel('Epsilon (Exploration Rate)')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Final AECI-Var (AI Echo Chamber)')
    plt.colorbar(im2, ax=ax)

    # Add values
    for i in range(len(learning_rate_values)):
        for j in range(len(epsilon_values)):
            ax.text(j, i, f'{aeci_matrix[i, j]:.2f}',
                   ha='center', va='center', color='black', fontweight='bold')

    # Variance across parameters (robustness check)
    ax = axes[1, 0]
    seci_variance = np.var(seci_matrix)
    aeci_variance = np.var(aeci_matrix)

    ax.bar(['SECI\nVariance', 'AECI\nVariance'],
           [seci_variance, aeci_variance],
           color=['#FF6B6B', '#4ECDC4'], edgecolor='black', linewidth=2)
    ax.set_ylabel('Variance Across Parameters')
    ax.set_title('Robustness Check: Lower = More Robust')
    ax.grid(True, axis='y', alpha=0.3)

    # Add interpretation text
    ax.text(0, seci_variance + 0.01, f'{seci_variance:.3f}',
           ha='center', va='bottom', fontweight='bold')
    ax.text(1, aeci_variance + 0.01, f'{aeci_variance:.3f}',
           ha='center', va='bottom', fontweight='bold')

    # Summary text
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = f"""
ROBUSTNESS ANALYSIS SUMMARY

Parameter Ranges Tested:
  Learning Rate: {learning_rate_values}
  Epsilon: {epsilon_values}

SECI Robustness:
  Mean: {np.mean(seci_matrix):.3f}
  Std: {np.std(seci_matrix):.3f}
  Range: [{np.min(seci_matrix):.3f}, {np.max(seci_matrix):.3f}]

AECI Robustness:
  Mean: {np.mean(aeci_matrix):.3f}
  Std: {np.std(aeci_matrix):.3f}
  Range: [{np.min(aeci_matrix):.3f}, {np.max(aeci_matrix):.3f}]

Interpretation:
  Low variance → Findings are robust
  High variance → Sensitive to parameters
    """

    ax.text(0.1, 0.5, summary_text,
           fontsize=10, family='monospace',
           verticalalignment='center')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "experiment_D_sensitivity_analysis.png"),
                dpi=300, bbox_inches='tight')
    plt.close()

    ##############################################
    # Final Summary
    ##############################################
    total_time = (time.time() - start_time) / 60

    print("\n" + "="*70)
    print("✓ ALL ESSENTIAL EXPERIMENTS COMPLETED!")
    print("="*70)
    print(f"Total runtime: {total_time:.1f} minutes ({total_time/60:.1f} hours)")
    print(f"\nResults saved to: {save_dir}")
    print("\nFiles created:")
    print(f"  ✓ results_experiment_A.pkl (Share Exploitative)")
    print(f"  ✓ results_experiment_B.pkl (AI Alignment)")
    print(f"  ✓ results_experiment_D.pkl (Q-Learning Sensitivity)")
    print("\nNext steps:")
    print("  1. Run the enhanced analysis script for SECI+AECI overlays")
    print("  2. Review plots in the results directory")
    print("  3. Optional: Run Experiment E (Noise/Rumors) for RQ5")
    print("="*70)
