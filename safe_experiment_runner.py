"""
SAFE Experiment Runner - Saves After EVERY Run
===============================================

This version saves results incrementally so you don't lose everything
if something crashes.

Add this to the beginning of the main section in DisasterAI_Model.py
to wrap experiments in try-except blocks.
"""

# REPLACE the main experiment code with this safer version:

if __name__ == "__main__":
    import time
    import traceback

    start_time = time.time()

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

    num_runs = 10

    ##############################################
    # Experiment A - WITH ERROR HANDLING
    ##############################################
    try:
        print("\n" + "="*70)
        print("EXPERIMENT A: Share Exploitative")
        print("="*70)

        share_values = [0.2, 0.5, 0.8]
        file_a_pkl = os.path.join(save_dir, "results_experiment_A.pkl")

        print(f"Starting Experiment A with {len(share_values)} values...")
        results_a = experiment_share_exploitative(base_params, share_values, num_runs)

        # SAVE IMMEDIATELY
        print(f"Saving to {file_a_pkl}...")
        with open(file_a_pkl, "wb") as f:
            pickle.dump(results_a, f)
        print(f"✓✓✓ EXPERIMENT A SAVED ✓✓✓")

        # Then plot (can fail without losing data)
        try:
            print("Generating plots...")
            for share in share_values:
                results_dict = results_a.get(share, {})
                if results_dict:
                    title_suffix = f"(Share={share})"
                    plot_simulation_overview(results_dict, title_suffix)
                    plot_echo_chamber_indices(results_dict, title_suffix)
        except Exception as plot_error:
            print(f"⚠ Plotting failed (but data is saved): {plot_error}")

    except Exception as e:
        print(f"✗✗✗ EXPERIMENT A FAILED ✗✗✗")
        print(f"Error: {e}")
        traceback.print_exc()

    ##############################################
    # Experiment B - WITH ERROR HANDLING
    ##############################################
    try:
        print("\n" + "="*70)
        print("EXPERIMENT B: AI Alignment")
        print("="*70)

        alignment_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        file_b_pkl = os.path.join(save_dir, "results_experiment_B.pkl")

        print(f"Starting Experiment B with {len(alignment_values)} values...")
        results_b = experiment_alignment_tipping_point(base_params, alignment_values, num_runs=10)

        # SAVE IMMEDIATELY
        print(f"Saving to {file_b_pkl}...")
        with open(file_b_pkl, "wb") as f:
            pickle.dump(results_b, f)
        print(f"✓✓✓ EXPERIMENT B SAVED ✓✓✓")

        # Then plot (can fail without losing data)
        try:
            print("Generating plots...")
            all_alignment_values = sorted(list(results_b.keys()))
            plot_final_echo_indices_vs_alignment(results_b, all_alignment_values)
        except Exception as plot_error:
            print(f"⚠ Plotting failed (but data is saved): {plot_error}")

    except Exception as e:
        print(f"✗✗✗ EXPERIMENT B FAILED ✗✗✗")
        print(f"Error: {e}")
        traceback.print_exc()

    ##############################################
    # Experiment D - WITH ERROR HANDLING
    ##############################################
    try:
        print("\n" + "="*70)
        print("EXPERIMENT D: Q-Learning Sensitivity")
        print("="*70)

        learning_rate_values = [0.05, 0.1, 0.15]
        epsilon_values = [0.2, 0.3, 0.4]
        file_d_pkl = os.path.join(save_dir, "results_experiment_D.pkl")

        print(f"Starting Experiment D with {len(learning_rate_values)}×{len(epsilon_values)} combinations...")
        results_d = experiment_learning_trust(base_params, learning_rate_values, epsilon_values, num_runs)

        # SAVE IMMEDIATELY
        print(f"Saving to {file_d_pkl}...")
        with open(file_d_pkl, "wb") as f:
            pickle.dump(results_d, f)
        print(f"✓✓✓ EXPERIMENT D SAVED ✓✓✓")

    except Exception as e:
        print(f"✗✗✗ EXPERIMENT D FAILED ✗✗✗")
        print(f"Error: {e}")
        traceback.print_exc()

    ##############################################
    # Summary
    ##############################################
    total_time = (time.time() - start_time) / 60
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*70)
    print(f"Total time: {total_time:.1f} minutes")
    print(f"Results saved to: {save_dir}")

    # Verify files exist
    print("\nVerifying saved files:")
    for exp, filename in [('A', 'results_experiment_A.pkl'),
                          ('B', 'results_experiment_B.pkl'),
                          ('D', 'results_experiment_D.pkl')]:
        filepath = os.path.join(save_dir, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024*1024)
            print(f"  ✓ Experiment {exp}: {filename} ({size_mb:.2f} MB)")
        else:
            print(f"  ✗ Experiment {exp}: NOT FOUND")

    print("="*70)
