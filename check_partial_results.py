#!/usr/bin/env python3
"""
Check what partial results exist from interrupted Colab run
"""

import os
import pickle
import numpy as np

# Check both possible locations
possible_dirs = [
    "agent_model_results",
    "/content/drive/MyDrive/DisasterAI_Results"
]

print("=== Checking for Partial Results ===\n")

for results_dir in possible_dirs:
    if not os.path.exists(results_dir):
        print(f"✗ {results_dir} - does not exist")
        continue

    print(f"✓ {results_dir} - found!")

    # List all pickle files
    pkl_files = [f for f in os.listdir(results_dir) if f.endswith('.pkl')]

    if not pkl_files:
        print(f"  No .pkl files found\n")
        continue

    print(f"  Found {len(pkl_files)} result file(s):\n")

    for pkl_file in sorted(pkl_files):
        filepath = os.path.join(results_dir, pkl_file)
        print(f"  📊 {pkl_file}")

        try:
            with open(filepath, 'rb') as f:
                results = pickle.load(f)

            # Analyze what's inside
            if isinstance(results, dict):
                print(f"     Dictionary with {len(results)} keys")

                # Check for experiment structure
                if any(isinstance(k, (int, float)) for k in results.keys()):
                    # Parameter sweep results
                    param_values = sorted([k for k in results.keys() if isinstance(k, (int, float))])
                    print(f"     Parameter values completed: {param_values}")

                    # Check each parameter's data quality
                    for param in param_values:
                        param_data = results[param]
                        if isinstance(param_data, dict):
                            # Check for key metrics
                            metrics = []
                            if 'aeci_variance' in param_data:
                                aeci_var = param_data['aeci_variance']
                                if isinstance(aeci_var, np.ndarray) and aeci_var.size > 0:
                                    metrics.append(f"AECI-Var: {aeci_var.shape}")
                            if 'aeci' in param_data:
                                aeci = param_data['aeci']
                                if isinstance(aeci, dict):
                                    metrics.append(f"AECI: exploit={aeci.get('exploit', 'N/A')}, explor={aeci.get('explor', 'N/A')}")
                            if 'belief_error' in param_data:
                                be = param_data['belief_error']
                                if isinstance(be, np.ndarray) and be.size > 0:
                                    metrics.append(f"Belief Error: {be.shape}")

                            if metrics:
                                print(f"       {param}: {', '.join(metrics)}")
                else:
                    # Single experiment results
                    print(f"     Keys: {list(results.keys())[:10]}...")

                    # Check for standard metrics
                    if 'aeci_variance' in results:
                        aeci_var = results['aeci_variance']
                        if isinstance(aeci_var, np.ndarray):
                            print(f"     AECI-Var shape: {aeci_var.shape}")
                            print(f"     AECI-Var range: [{np.nanmin(aeci_var):.4f}, {np.nanmax(aeci_var):.4f}]")

                    if 'aeci' in results:
                        print(f"     AECI data: {type(results['aeci'])}")

            print()

        except Exception as e:
            print(f"     ✗ Error loading: {e}\n")

print("\n=== Recommendations ===")
print("If you have completed parameter values, we can:")
print("1. Generate plots for the completed data")
print("2. Show you which parameter values need re-running")
print("3. Create a resume script that only runs missing values")
