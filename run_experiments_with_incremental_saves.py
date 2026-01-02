"""
TRULY INCREMENTAL Experiment Runner - USING EXISTING WORKING FUNCTIONS
=======================================================================

Uses the proven aggregate_simulation_results() function from the model,
but saves after each parameter value completes.
"""

import os
import sys
import pickle
import time

# Import from main model
sys.path.insert(0, '/content/DisasterAI')
from DisasterAI_Model import aggregate_simulation_results, DisasterModel

# Configuration
try:
    from google.colab import drive
    # Check if already mounted, if not mount it
    if not os.path.exists('/content/drive/MyDrive'):
        drive.mount('/content/drive')
    save_dir = "/content/drive/MyDrive/DisasterAI_results"
    IN_COLAB = True
except ImportError:
    # Not in Colab
    save_dir = "DisasterAI_results"
    IN_COLAB = False

os.makedirs(save_dir, exist_ok=True)

print("="*70)
print("SAVE LOCATION VERIFICATION")
print("="*70)
print(f"IN_COLAB: {IN_COLAB}")
print(f"Save directory: {save_dir}")
print(f"Directory exists: {os.path.exists(save_dir)}")
print(f"Directory is writable: {os.access(save_dir, os.W_OK)}")
if IN_COLAB:
    drive_mounted = os.path.exists("/content/drive/MyDrive")
    print(f"Drive mounted: {drive_mounted}")
    if not drive_mounted:
        print("\n⚠⚠⚠ DRIVE NOT MOUNTED - FILES WILL BE LOST! ⚠⚠⚠\n")
        sys.exit(1)
print("="*70)
print(f"✓ This version uses EXISTING WORKING FUNCTIONS")
print(f"✓ Saves after EACH parameter value")
print("="*70)

# Base parameters (from original code)
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
    "lambda_parameter": 0.5,
    "learning_rate": 0.05,
    "epsilon": 0.2,
    "ticks": 150,
    "rumor_probability": 0.7,
    "rumor_intensity": 2.0
}

num_runs = 10

#########################################
# Experiment B - AI Alignment
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
        # Prepare params for this alignment value
        params = base_params.copy()
        params["ai_alignment_level"] = align_val

        # Use the EXISTING working function
        result = aggregate_simulation_results(num_runs, params)
        results_b[align_val] = result

        # SAVE IMMEDIATELY (both progress and main)
        print(f"    Saving to: {file_b_pkl}")
        with open(file_b_progress, 'wb') as f:
            pickle.dump(results_b, f)
        with open(file_b_pkl, 'wb') as f:
            pickle.dump(results_b, f)

        # VERIFY FILES EXIST
        if os.path.exists(file_b_pkl):
            size = os.path.getsize(file_b_pkl) / (1024*1024)
            print(f"    ✓ SAVED (alignment={align_val:.2f}) - {size:.2f} MB")
        else:
            print(f"    ✗ SAVE FAILED - FILE DOES NOT EXIST!")
            raise Exception(f"File not found after save: {file_b_pkl}")

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
# Experiment D - Q-Learning Sensitivity
#########################################

print("\n▶ EXPERIMENT D: Q-Learning Sensitivity (INCREMENTAL SAVES)")
learning_rates = [0.05, 0.1, 0.15]
epsilons = [0.2, 0.3, 0.4]
file_d_pkl = os.path.join(save_dir, "results_experiment_D.pkl")
file_d_progress = os.path.join(save_dir, "results_experiment_D_PROGRESS.pkl")

# Try to resume from progress file
if os.path.exists(file_d_progress):
    print(f"  Found progress file - resuming!")
    with open(file_d_progress, 'rb') as f:
        results_d = pickle.load(f)
    completed_d = list(results_d.keys())
    print(f"  Already completed: {completed_d}")
else:
    results_d = {}
    completed_d = []

start_d = time.time()
param_combos = [(lr, eps) for lr in learning_rates for eps in epsilons]

for i, (lr, eps) in enumerate(param_combos):
    key = (lr, eps)
    if key in completed_d:
        print(f"  [{i+1}/{len(param_combos)}] LR={lr}, ε={eps} - SKIPPING (already done)")
        continue

    print(f"\n  [{i+1}/{len(param_combos)}] Starting LR={lr}, ε={eps}...")

    try:
        # Prepare params for this combination
        params = base_params.copy()
        params["learning_rate"] = lr
        params["epsilon"] = eps

        # Use the EXISTING working function
        result = aggregate_simulation_results(num_runs, params)
        results_d[key] = result

        # SAVE IMMEDIATELY (both progress and main)
        print(f"    Saving to: {file_d_pkl}")
        with open(file_d_progress, 'wb') as f:
            pickle.dump(results_d, f)
        with open(file_d_pkl, 'wb') as f:
            pickle.dump(results_d, f)

        # VERIFY FILES EXIST
        if os.path.exists(file_d_pkl):
            size = os.path.getsize(file_d_pkl) / (1024*1024)
            print(f"    ✓ SAVED (LR={lr}, ε={eps}) - {size:.2f} MB")
        else:
            print(f"    ✗ SAVE FAILED - FILE DOES NOT EXIST!")
            raise Exception(f"File not found after save: {file_d_pkl}")

    except Exception as e:
        print(f"\n    ✗ FAILED at LR={lr}, ε={eps}: {e}")
        import traceback
        traceback.print_exc()
        print(f"    Partial results saved. You can resume from here.")
        break

elapsed_d = (time.time() - start_d) / 60
print(f"\n✓ EXPERIMENT D: {len(results_d)}/{len(param_combos)} combinations complete ({elapsed_d:.1f} min)")
print(f"✓ Saved to: {file_d_pkl}")

# Clean up progress file if fully complete
if len(results_d) == len(param_combos):
    if os.path.exists(file_d_progress):
        os.remove(file_d_progress)
    print("✓ Fully complete - removed progress file")

print("\n" + "="*70)
print("EXPERIMENTS COMPLETE")
print("="*70)
print(f"Results in: {save_dir}")
print("If anything crashed, just re-run this script - it will resume!")
print("="*70)
