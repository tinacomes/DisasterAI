"""
Experiment E: Noise & Rumor Sensitivity Analysis
=================================================

Tests RQ5: How do findings depend on uncertainty, noise, and rumors?

This experiment varies:
1. Rumor intensity (strength of false information)
2. Initial noise levels (uncertainty in initial beliefs)

To add to DisasterAI_Model.py:
1. Add the experiment function (below)
2. Add the main experiment code (at bottom of main section)
"""

#########################################
# ADD THIS FUNCTION TO DisasterAI_Model.py
# (After the other experiment_ functions, around line 3850)
#########################################

def experiment_noise_rumors(base_params, rumor_intensities, noise_levels, num_runs=10):
    """
    Experiment E: Test sensitivity to noise and rumors (RQ5).

    Args:
        base_params: Base parameter dictionary
        rumor_intensities: List of rumor intensity values [0.0, 0.3, 0.6]
        noise_levels: List of noise probability values [0.2, 0.4, 0.6]
        num_runs: Number of simulation runs per combination

    Returns:
        results: Dict mapping (rumor_intensity, noise_level) → aggregated metrics
    """
    import copy
    import numpy as np

    results = {}
    total_combos = len(rumor_intensities) * len(noise_levels)
    current_combo = 0

    for rumor_intensity in rumor_intensities:
        for noise_level in noise_levels:
            current_combo += 1
            print(f"\n--- Combo {current_combo}/{total_combos}: Rumor={rumor_intensity:.2f}, Noise={noise_level:.2f} ---")

            # Prepare params
            params_copy = copy.deepcopy(base_params)

            # Set moderate AI alignment for testing
            params_copy["ai_alignment_level"] = 0.5

            # Create rumor configuration
            # Rumor at center of grid with varying intensity
            width = params_copy["width"]
            height = params_copy["height"]
            rumor_config = (
                (width // 2, height // 2),  # Center of grid
                rumor_intensity,             # Intensity (0-1)
                0.8,                        # High confidence in rumor
                width // 4                  # Radius (quarter of grid)
            )

            # Run simulations
            run_data = []
            for run in range(num_runs):
                print(f"  Run {run + 1}/{num_runs}...", end=" ")

                # Create model
                model = DisasterModel(
                    width=params_copy["width"],
                    height=params_copy["height"],
                    number_of_humans=params_copy["number_of_humans"],
                    share_exploitative=params_copy["share_exploitative"],
                    share_of_disaster=params_copy["share_of_disaster"],
                    initial_trust=params_copy["initial_trust"],
                    initial_ai_trust=params_copy["initial_ai_trust"],
                    share_confirming=params_copy["share_confirming"],
                    disaster_dynamics=params_copy["disaster_dynamics"],
                    shock_probability=params_copy["shock_probability"],
                    shock_magnitude=params_copy["shock_magnitude"],
                    ai_alignment_level=params_copy["ai_alignment_level"],
                    num_ai=params_copy["num_ai"],
                    trust_update_mode=params_copy["trust_update_mode"],
                    debug_mode=params_copy.get("debug_mode", False)
                )

                # INJECT RUMOR into agents
                for agent in model.humans.values():
                    agent.initialize_beliefs(assigned_rumor=rumor_config)

                    # MODIFY noise level in belief initialization
                    # This is a hack - ideally we'd pass noise_level to initialize_beliefs
                    # For now, we re-randomize some beliefs with the new noise level
                    if noise_level > 0.4:  # If higher noise than default
                        for cell in list(agent.beliefs.keys()):
                            if random.random() < (noise_level - 0.4):
                                # Add extra noise
                                current_level = agent.beliefs[cell]['level']
                                noise = random.choice([-2, -1, 0, 1, 2])
                                agent.beliefs[cell]['level'] = max(0, min(5, current_level + noise))
                                # Reduce confidence
                                agent.beliefs[cell]['confidence'] *= 0.8

                # Run simulation
                max_ticks = params_copy["max_ticks"]
                for tick in range(max_ticks):
                    model.step()

                # Collect data
                run_data.append({
                    'seci': np.array(model.seci_data),
                    'aeci_variance': np.array(model.aeci_variance_data),
                    'trust_stats': np.array(model.trust_data),
                    'info_diversity': np.array(model.info_diversity_data),
                    'correct_tokens': model.correct_tokens,
                    'incorrect_tokens': model.incorrect_tokens
                })

                print("Done")

                del model
                import gc
                gc.collect()

            # Aggregate results
            print(f"  Aggregating {num_runs} runs...")
            aggregated = aggregate_runs(run_data)
            results[(rumor_intensity, noise_level)] = aggregated

            print(f"  ✓ Completed rumor={rumor_intensity:.2f}, noise={noise_level:.2f}")

    return results


#########################################
# ADD THIS TO MAIN SECTION OF DisasterAI_Model.py
# (After Experiment D, around line 7460)
#########################################

"""
    ##############################################
    # Experiment E: Noise & Rumor Sensitivity
    ##############################################
    print("\n=== RUNNING EXPERIMENT E: Noise & Rumor Sensitivity ===")
    rumor_intensities = [0.0, 0.3, 0.6]  # No rumor, medium, high
    noise_levels = [0.2, 0.4, 0.6]       # Low, medium, high noise
    file_e_pkl = os.path.join(save_dir, "results_experiment_E.pkl")

    print(f"Testing {len(rumor_intensities)} rumor levels × {len(noise_levels)} noise levels...")
    results_e = experiment_noise_rumors(base_params, rumor_intensities, noise_levels, num_runs)

    # Save results
    with open(file_e_pkl, "wb") as f:
        pickle.dump(results_e, f)
    print(f"✓ Experiment E saved to: {file_e_pkl}")

    # Generate heatmap visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Experiment E: Sensitivity to Noise & Rumors (RQ5)",
                 fontsize=16, fontweight='bold')

    # Create matrices for heatmaps
    seci_matrix = np.zeros((len(rumor_intensities), len(noise_levels)))
    aeci_matrix = np.zeros((len(rumor_intensities), len(noise_levels)))
    performance_matrix = np.zeros((len(rumor_intensities), len(noise_levels)))

    for i, rumor in enumerate(rumor_intensities):
        for j, noise in enumerate(noise_levels):
            res = results_e.get((rumor, noise), {})

            # Extract final SECI
            seci_data = res.get("seci", np.array([]))
            if seci_data.size > 0:
                seci_matrix[i, j] = np.mean(seci_data[:, -1, 1])  # Exploitative

            # Extract final AECI
            aeci_data = res.get("aeci_variance", np.array([]))
            if aeci_data.size > 0:
                aeci_matrix[i, j] = np.mean(aeci_data[:, -1, 1])

            # Extract performance
            correct = res.get("correct_tokens", {}).get("exploitative", 0)
            incorrect = res.get("incorrect_tokens", {}).get("exploitative", 0)
            total = correct + incorrect
            if total > 0:
                performance_matrix[i, j] = correct / total

    # Plot SECI heatmap
    ax = axes[0, 0]
    im1 = ax.imshow(seci_matrix, cmap='RdYlGn_r', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(len(noise_levels)))
    ax.set_yticks(range(len(rumor_intensities)))
    ax.set_xticklabels([f'{n:.1f}' for n in noise_levels])
    ax.set_yticklabels([f'{r:.1f}' for r in rumor_intensities])
    ax.set_xlabel('Noise Level')
    ax.set_ylabel('Rumor Intensity')
    ax.set_title('Final SECI (Social Echo Chamber)')
    plt.colorbar(im1, ax=ax)

    # Add values
    for i in range(len(rumor_intensities)):
        for j in range(len(noise_levels)):
            ax.text(j, i, f'{seci_matrix[i, j]:.2f}',
                   ha='center', va='center', color='black', fontweight='bold')

    # Plot AECI heatmap
    ax = axes[0, 1]
    im2 = ax.imshow(aeci_matrix, cmap='RdYlGn_r', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(len(noise_levels)))
    ax.set_yticks(range(len(rumor_intensities)))
    ax.set_xticklabels([f'{n:.1f}' for n in noise_levels])
    ax.set_yticklabels([f'{r:.1f}' for r in rumor_intensities])
    ax.set_xlabel('Noise Level')
    ax.set_ylabel('Rumor Intensity')
    ax.set_title('Final AECI-Var (AI Echo Chamber)')
    plt.colorbar(im2, ax=ax)

    # Add values
    for i in range(len(rumor_intensities)):
        for j in range(len(noise_levels)):
            ax.text(j, i, f'{aeci_matrix[i, j]:.2f}',
                   ha='center', va='center', color='black', fontweight='bold')

    # Plot performance heatmap
    ax = axes[1, 0]
    im3 = ax.imshow(performance_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(noise_levels)))
    ax.set_yticks(range(len(rumor_intensities)))
    ax.set_xticklabels([f'{n:.1f}' for n in noise_levels])
    ax.set_yticklabels([f'{r:.1f}' for r in rumor_intensities])
    ax.set_xlabel('Noise Level')
    ax.set_ylabel('Rumor Intensity')
    ax.set_title('Accuracy (Correct Tokens %)')
    plt.colorbar(im3, ax=ax)

    # Add values
    for i in range(len(rumor_intensities)):
        for j in range(len(noise_levels)):
            ax.text(j, i, f'{performance_matrix[i, j]:.2f}',
                   ha='center', va='center', color='black', fontweight='bold')

    # Summary text
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = f'''
EXPERIMENT E SUMMARY

Rumor Intensities: {rumor_intensities}
Noise Levels: {noise_levels}

Key Findings:

SECI Robustness:
  Mean: {np.mean(seci_matrix):.3f}
  Std: {np.std(seci_matrix):.3f}

AECI Robustness:
  Mean: {np.mean(aeci_matrix):.3f}
  Std: {np.std(aeci_matrix):.3f}

Performance Impact:
  Best: {np.max(performance_matrix):.3f}
  Worst: {np.min(performance_matrix):.3f}
  Drop: {(np.max(performance_matrix) - np.min(performance_matrix)):.3f}

Interpretation:
  Low variance → Robust to noise/rumors
  High variance → Sensitive to environment
    '''

    ax.text(0.1, 0.5, summary_text,
           fontsize=10, family='monospace',
           verticalalignment='center')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "experiment_E_noise_rumors.png"),
                dpi=300, bbox_inches='tight')
    plt.close()

    print("=== EXPERIMENT E COMPLETED ===")
"""

#########################################
# END OF CODE TO ADD
#########################################

print("""
EXPERIMENT E - IMPLEMENTATION GUIDE
====================================

To add Experiment E to your model:

1. ADD THE FUNCTION:
   - Copy the experiment_noise_rumors() function above
   - Paste it in DisasterAI_Model.py after experiment_learning_trust() (around line 3850)

2. ADD THE MAIN CODE:
   - Copy the code between the triple quotes above
   - Paste it in the main section after Experiment D (around line 7460)
   - Remove the triple quotes when pasting!

3. RUNTIME:
   - 3 rumor levels × 3 noise levels = 9 combinations
   - 10 runs each = 90 simulations
   - Estimated time: ~60-90 minutes

4. WHAT IT TESTS (RQ5):
   - Does AI help under high uncertainty (noise)?
   - Do rumors prevent AI from breaking bubbles?
   - Is the system robust to misinformation?

This will create: results_experiment_E.pkl
""")
