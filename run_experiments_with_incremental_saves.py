"""
TRULY INCREMENTAL Experiment Runner
====================================

This version saves after EVERY SINGLE PARAMETER VALUE.
If it crashes, you only lose the current parameter, not everything.

This is what should have been done from the start. I apologize.
"""

import os
import sys
import pickle
import copy
import time

# Import from main model
sys.path.insert(0, '/content/DisasterAI')
from DisasterAI_Model import *

# Configuration
try:
    from google.colab import drive
    drive.mount('/content/drive')
    save_dir = "/content/drive/MyDrive/DisasterAI_results"
    IN_COLAB = True
except:
    save_dir = "DisasterAI_results"
    IN_COLAB = False

os.makedirs(save_dir, exist_ok=True)

print(f"✓ Save directory: {save_dir}")
print(f"✓ This version saves after EACH parameter value")
print("="*70)

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

#########################################
# Experiment B - WITH INCREMENTAL SAVES
#########################################

print("\n▶ EXPERIMENT B: AI Alignment (INCREMENTAL SAVES)")
alignment_values = [0.0, 0.25, 0.5, 0.75, 1.0]
file_b_pkl = os.path.join(save_dir, "results_experiment_B.pkl")
file_b_progress = os.path.join(save_dir, "results_experiment_B_PROGRESS.pkl")

# Try to resume from progress file
if os.path.exists(file_b_progress):
    print(f"  Found progress file - resuming!")
    with open(file_b_progress, 'rb') as f:
        results_b = pickle.load(f)
    completed = list(results_b.keys())
    print(f"  Already completed: {completed}")
else:
    results_b = {}
    completed = []

start_b = time.time()

for i, align_val in enumerate(alignment_values):
    if align_val in completed:
        print(f"  [{i+1}/{len(alignment_values)}] Alignment={align_val:.2f} - SKIPPING (already done)")
        continue

    print(f"\n  [{i+1}/{len(alignment_values)}] Starting Alignment={align_val:.2f}...")

    try:
        # Prepare params
        params_copy = copy.deepcopy(base_params)
        params_copy["ai_alignment_level"] = align_val

        # Run multiple runs for this parameter
        run_data = []
        for run in range(num_runs):
            print(f"    Run {run+1}/{num_runs}...", end=" ", flush=True)

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

            for tick in range(params_copy["max_ticks"]):
                model.step()

            run_data.append({
                'seci': np.array(model.seci_data),
                'aeci': np.array(model.aeci_data) if hasattr(model, 'aeci_data') else np.array([]),
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

        # Aggregate
        print(f"    Aggregating {num_runs} runs...")
        aggregated = aggregate_runs(run_data)
        results_b[align_val] = aggregated

        # SAVE IMMEDIATELY (both progress and main)
        with open(file_b_progress, 'wb') as f:
            pickle.dump(results_b, f)
        with open(file_b_pkl, 'wb') as f:
            pickle.dump(results_b, f)

        print(f"    ✓ SAVED (alignment={align_val:.2f})")

    except Exception as e:
        print(f"\n    ✗ FAILED at alignment={align_val:.2f}: {e}")
        import traceback
        traceback.print_exc()
        print(f"    Partial results saved. You can resume from here.")
        break

elapsed_b = (time.time() - start_b) / 60
print(f"\n✓ EXPERIMENT B: {len(results_b)}/{len(alignment_values)} parameters complete ({elapsed_b:.1f} min)")
print(f"✓ Saved to: {file_b_pkl}")

# Clean up progress file if fully complete
if len(results_b) == len(alignment_values):
    if os.path.exists(file_b_progress):
        os.remove(file_b_progress)
    print("✓ Fully complete - removed progress file")

#########################################
# Experiment D - WITH INCREMENTAL SAVES
#########################################

print("\n▶ EXPERIMENT D: Q-Learning Sensitivity (INCREMENTAL SAVES)")
learning_rate_values = [0.05, 0.1, 0.15]
epsilon_values = [0.2, 0.3, 0.4]
file_d_pkl = os.path.join(save_dir, "results_experiment_D.pkl")
file_d_progress = os.path.join(save_dir, "results_experiment_D_PROGRESS.pkl")

# Try to resume
if os.path.exists(file_d_progress):
    print(f"  Found progress file - resuming!")
    with open(file_d_progress, 'rb') as f:
        results_d = pickle.load(f)
    completed_d = list(results_d.keys())
    print(f"  Already completed: {len(completed_d)} combinations")
else:
    results_d = {}
    completed_d = []

start_d = time.time()
total_combos = len(learning_rate_values) * len(epsilon_values)
current_combo = 0

for lr in learning_rate_values:
    for eps in epsilon_values:
        current_combo += 1

        if (lr, eps) in completed_d:
            print(f"  [{current_combo}/{total_combos}] LR={lr:.2f}, ε={eps:.1f} - SKIPPING")
            continue

        print(f"\n  [{current_combo}/{total_combos}] Starting LR={lr:.2f}, ε={eps:.1f}...")

        try:
            params_copy = copy.deepcopy(base_params)
            params_copy["learning_rate"] = lr
            params_copy["epsilon"] = eps

            run_data = []
            for run in range(num_runs):
                print(f"    Run {run+1}/{num_runs}...", end=" ", flush=True)

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

                # Set learning parameters on agents
                for agent in model.humans.values():
                    agent.learning_rate = lr
                    agent.epsilon = eps

                for tick in range(params_copy["max_ticks"]):
                    model.step()

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

            # Aggregate
            print(f"    Aggregating {num_runs} runs...")
            aggregated = aggregate_runs(run_data)
            results_d[(lr, eps)] = aggregated

            # SAVE IMMEDIATELY
            with open(file_d_progress, 'wb') as f:
                pickle.dump(results_d, f)
            with open(file_d_pkl, 'wb') as f:
                pickle.dump(results_d, f)

            print(f"    ✓ SAVED (LR={lr:.2f}, ε={eps:.1f})")

        except Exception as e:
            print(f"\n    ✗ FAILED at LR={lr:.2f}, ε={eps:.1f}: {e}")
            import traceback
            traceback.print_exc()
            print(f"    Partial results saved. You can resume from here.")
            break

elapsed_d = (time.time() - start_d) / 60
print(f"\n✓ EXPERIMENT D: {len(results_d)}/{total_combos} combinations complete ({elapsed_d:.1f} min)")
print(f"✓ Saved to: {file_d_pkl}")

# Clean up progress file if fully complete
if len(results_d) == total_combos:
    if os.path.exists(file_d_progress):
        os.remove(file_d_progress)
    print("✓ Fully complete - removed progress file")

print("\n" + "="*70)
print("EXPERIMENTS COMPLETE")
print("="*70)
print(f"Results in: {save_dir}")
print("If anything crashed, just re-run this script - it will resume!")
print("="*70)
