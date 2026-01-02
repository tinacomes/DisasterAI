"""
Diagnostic script to inspect pickle file structure
"""
import pickle
import numpy as np
import os

RESULTS_DIR = "/content/drive/MyDrive/agent_model_results"

def inspect_pickle(pkl_path):
    """Thoroughly inspect a pickle file structure."""
    print(f"\n{'='*70}")
    print(f"INSPECTING: {pkl_path}")
    print('='*70)

    if not os.path.exists(pkl_path):
        print(f"⚠ File not found!")
        return

    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        print(f"✓ Loaded successfully")
        print(f"Type: {type(data)}")

        if isinstance(data, dict):
            print(f"Dictionary with {len(data)} keys")
            print(f"Keys (parameter values): {sorted(list(data.keys()))}")

            # Inspect first parameter value
            first_key = sorted(list(data.keys()))[0]
            print(f"\n--- Inspecting data for parameter = {first_key} ---")

            param_data = data[first_key]
            print(f"Type: {type(param_data)}")

            if isinstance(param_data, dict):
                print(f"Dictionary with {len(param_data)} keys")
                print(f"Keys (metrics): {list(param_data.keys())}")

                # Inspect each metric
                for metric_name in ['seci', 'aeci_variance', 'aeci', 'trust_stats', 'info_diversity']:
                    if metric_name in param_data:
                        metric_data = param_data[metric_name]
                        print(f"\n  {metric_name}:")
                        print(f"    Type: {type(metric_data)}")

                        if isinstance(metric_data, np.ndarray):
                            print(f"    Shape: {metric_data.shape}")
                            print(f"    Dtype: {metric_data.dtype}")

                            # Show sample data
                            if metric_data.size > 0:
                                if metric_data.ndim == 3:
                                    print(f"    Format: (runs={metric_data.shape[0]}, ticks={metric_data.shape[1]}, data={metric_data.shape[2]})")
                                    print(f"    Sample (run 0, first 3 ticks):")
                                    print(f"      {metric_data[0, :3, :]}")
                                elif metric_data.ndim == 2:
                                    print(f"    Format: (runs={metric_data.shape[0]}, ticks={metric_data.shape[1]})")
                                    print(f"    Sample (run 0, first 5 values):")
                                    print(f"      {metric_data[0, :5]}")
                                elif metric_data.ndim == 1:
                                    print(f"    Sample (first 5 values):")
                                    print(f"      {metric_data[:5]}")
                        elif isinstance(metric_data, list):
                            print(f"    List with {len(metric_data)} items")
                            if len(metric_data) > 0:
                                print(f"    First item type: {type(metric_data[0])}")
                                print(f"    Sample: {metric_data[:2]}")
                        else:
                            print(f"    Value: {metric_data}")
                    else:
                        print(f"\n  {metric_name}: NOT FOUND")

        elif isinstance(data, np.ndarray):
            print(f"NumPy array with shape: {data.shape}")

        else:
            print(f"Data: {data}")

    except Exception as e:
        print(f"⚠ Error loading: {e}")
        import traceback
        traceback.print_exc()

# Inspect both experiment files
print("="*70)
print("PICKLE FILE STRUCTURE INSPECTION")
print("="*70)

inspect_pickle(os.path.join(RESULTS_DIR, "results_experiment_A.pkl"))
inspect_pickle(os.path.join(RESULTS_DIR, "results_experiment_B.pkl"))

# Also check the other location
print("\n\n" + "="*70)
print("CHECKING OTHER RESULT LOCATIONS")
print("="*70)

other_files = [
    "/content/drive/MyDrive/DisasterAI_results/results_experiment_B_bis_FULL.pkl"
]

for pkl_path in other_files:
    if os.path.exists(pkl_path):
        inspect_pickle(pkl_path)
