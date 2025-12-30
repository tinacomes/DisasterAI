#!/usr/bin/env python3
"""
Quick analysis of existing results to check AECI-Var values
"""

import pickle
import numpy as np

results_dir = "agent_model_results"

print("=== Quick AECI-Var Check ===\n")

# Check Experiment B results
print("📊 Experiment B (AI Alignment Tipping Point)")
with open(f"{results_dir}/results_AI_Alignment_Tipping_Point.pkl", 'rb') as f:
    exp_b = pickle.load(f)

for alignment in sorted([k for k in exp_b.keys() if isinstance(k, (int, float))]):
    aeci_var_data = exp_b[alignment].get('aeci_variance')

    if aeci_var_data is not None and isinstance(aeci_var_data, np.ndarray):
        # Extract just the AECI-Var values (column 1)
        aeci_var_values = aeci_var_data[:, :, 1]

        # Get final tick values across all runs
        final_tick_values = aeci_var_values[:, -1]

        print(f"\n  Alignment = {alignment}")
        print(f"    Final tick AECI-Var: {final_tick_values}")
        print(f"    Range: [{np.min(final_tick_values):.4f}, {np.max(final_tick_values):.4f}]")
        print(f"    Mean: {np.mean(final_tick_values):.4f}")
        print(f"    All zeros? {np.all(final_tick_values == 0)}")

print("\n" + "="*50)
print("\n📊 Experiment A (Share Values)")
with open(f"{results_dir}/results_experiment_A.pkl", 'rb') as f:
    exp_a = pickle.load(f)

for share in sorted([k for k in exp_a.keys() if isinstance(k, (int, float))]):
    aeci_var_data = exp_a[share].get('aeci_variance')

    if aeci_var_data is not None and isinstance(aeci_var_data, np.ndarray):
        aeci_var_values = aeci_var_data[:, :, 1]
        final_tick_values = aeci_var_values[:, -1]

        print(f"\n  Share = {share}")
        print(f"    Final tick AECI-Var: {final_tick_values}")
        print(f"    Range: [{np.min(final_tick_values):.4f}, {np.max(final_tick_values):.4f}]")
        print(f"    Mean: {np.mean(final_tick_values):.4f}")
        print(f"    All zeros? {np.all(final_tick_values == 0)}")

print("\n" + "="*50)
print("\n💡 Verdict:")
print("   If all AECI-Var values are zero, these results used the old threshold.")
print("   You'll need to re-run with the new threshold (min_calls_threshold=5)")
print("   to see real AI echo chamber effects.")
print("\n   However, all OTHER metrics (SECI, AECI ratio, trust, etc.) are valid!")
print("   We can still generate those plots if you want to see them.")
