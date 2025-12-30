#!/usr/bin/env python3
"""
Quick test run of Experiment B with 5 runs (instead of 10)
Use this to verify AECI-Var threshold fix is working before full run
"""

import os
import pickle

# Set environment variable for save directory if in Colab
try:
    from google.colab import drive
    IN_COLAB = True
    # Mount drive if not already mounted
    if not os.path.exists('/content/drive/MyDrive'):
        drive.mount('/content/drive')
    save_dir = "/content/drive/MyDrive/DisasterAI_Results"
except:
    IN_COLAB = False
    save_dir = "agent_model_results"

os.environ['SAVE_DIR'] = save_dir
os.makedirs(save_dir, exist_ok=True)

print(f"🎯 Quick Test: Experiment B with 5 runs")
print(f"📁 Results will be saved to: {save_dir}\n")

# Import after setting environment variable
from DisasterAI_Model import (
    experiment_alignment_tipping_point,
    plot_final_echo_indices_vs_alignment,
    plot_average_performance_vs_alignment,
    plot_summary_echo_indices_vs_alignment,
    plot_summary_performance_vs_alignment,
    plot_phase_diagram_bubbles,
    plot_tipping_point_waterfall
)

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
    "lambda_parameter": 0.5,
    "learning_rate": 0.05,
    "epsilon": 0.2,
    "ticks": 150,
    "rumor_probability": 0.7,
    "rumor_intensity": 2.0,
    "rumor_confidence": 0.75,
    "rumor_radius_factor": 0.9,
    "min_rumor_separation_factor": 0.5,
    "exploit_trust_lr": 0.03,
    "explor_trust_lr": 0.08,
    "exploit_friend_bias": 0.1,
    "exploit_self_bias": 0.1
}

# Experiment parameters
alignment_values = [0.0, 0.25, 0.5, 0.75, 0.95]
num_runs = 5  # Quick test with fewer runs
param_name = "AI Alignment Tipping Point"

print(f"⚙️  Running {param_name} with {num_runs} runs...")
print(f"⏱️  Estimated time: ~8-10 minutes\n")

# Run experiment
results_b = experiment_alignment_tipping_point(base_params, alignment_values, num_runs=num_runs)

# Save results
file_b_pkl = os.path.join(save_dir, "results_AI_Alignment_Quick_Test.pkl")
with open(file_b_pkl, "wb") as f:
    pickle.dump(results_b, f)
print(f"✅ Results saved to: {file_b_pkl}\n")

# Get all alignment values
all_alignment_values = sorted(list(results_b.keys()))

# Generate plots
print("📊 Generating plots...\n")

print("  → Final echo indices vs alignment")
plot_final_echo_indices_vs_alignment(results_b, all_alignment_values, title_suffix="Quick Test (5 runs)")

print("  → Average performance vs alignment")
plot_average_performance_vs_alignment(results_b, all_alignment_values, title_suffix="Quick Test (5 runs)")

print("  → Boxplot summaries (echo indices)")
plot_summary_echo_indices_vs_alignment(results_b, all_alignment_values, title_suffix="Quick Test (5 runs)")

print("  → Boxplot summaries (performance)")
plot_summary_performance_vs_alignment(results_b, all_alignment_values, title_suffix="Quick Test (5 runs)")

print("  → Phase diagram bubbles")
plot_phase_diagram_bubbles(results_b, all_alignment_values, param_name="AI Alignment")

print("  → Tipping point waterfall")
plot_tipping_point_waterfall(results_b, all_alignment_values, param_name="AI Alignment")

print("\n" + "="*60)
print("✅ QUICK TEST COMPLETE!")
print("="*60)
print("\n📋 Next Steps:")
print("   1. Check the console output above for threshold diagnostics")
print("      Look for: '≥3 total queries: X agents qualify'")
print("   2. Check the plots - AECI-Var should have non-zero values")
print("   3. If AECI-Var is working, run full 10-run experiment:")
print("      %run DisasterAI_Model.py")
print(f"\n📁 All results saved to: {save_dir}")
