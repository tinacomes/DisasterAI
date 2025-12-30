#!/usr/bin/env python3
"""
Reload existing results and regenerate plots with fixed visualization code
No need to rerun simulations - just replot!
"""

import os
import pickle
import sys

# Detect Colab and set paths
try:
    from google.colab import drive
    save_dir = "/content/drive/MyDrive/DisasterAI_Results"
except:
    save_dir = "agent_model_results"

print(f"📁 Loading results from: {save_dir}\n")

# Import plotting functions from main file
from DisasterAI_Model import (
    plot_final_echo_indices_vs_alignment,
    plot_average_performance_vs_alignment,
    plot_summary_echo_indices_vs_alignment,
    plot_summary_performance_vs_alignment,
    plot_phase_diagram_bubbles,
    plot_tipping_point_waterfall
)

# Load Experiment B results
file_b_pkl = os.path.join(save_dir, "results_AI_Alignment_Tipping_Point.pkl")

if not os.path.exists(file_b_pkl):
    print(f"❌ Results file not found: {file_b_pkl}")
    print("\nAvailable files:")
    if os.path.exists(save_dir):
        for f in os.listdir(save_dir):
            if f.endswith('.pkl'):
                print(f"  - {f}")
    else:
        print(f"  Directory {save_dir} does not exist!")
    sys.exit(1)

print(f"✅ Found results file: {file_b_pkl}")
print("📊 Loading data...")

with open(file_b_pkl, 'rb') as f:
    results_b = pickle.load(f)

# Get alignment values
all_alignment_values = sorted([k for k in results_b.keys() if isinstance(k, (int, float))])

print(f"✅ Loaded results for alignment values: {all_alignment_values}")
print(f"   ({len(all_alignment_values)} parameter values)\n")

# Regenerate all plots with FIXED code
print("🎨 Regenerating plots with fixed code...\n")

print("  → Final echo indices vs alignment")
plot_final_echo_indices_vs_alignment(results_b, all_alignment_values, title_suffix="(Replotted)")

print("  → Average performance vs alignment")
plot_average_performance_vs_alignment(results_b, all_alignment_values, title_suffix="(Replotted)")

print("  → Boxplot summaries (echo indices)")
plot_summary_echo_indices_vs_alignment(results_b, all_alignment_values, title_suffix="(Replotted)")

print("  → Boxplot summaries (performance)")
plot_summary_performance_vs_alignment(results_b, all_alignment_values, title_suffix="(Replotted)")

print("  → Phase diagram bubbles")
plot_phase_diagram_bubbles(results_b, all_alignment_values, param_name="AI Alignment")

print("  → Tipping point waterfall")
plot_tipping_point_waterfall(results_b, all_alignment_values, param_name="AI Alignment")

print("\n" + "="*60)
print("✅ ALL PLOTS REGENERATED WITH FIXES!")
print("="*60)
print("\n📋 What should be fixed now:")
print("   ✅ Phase diagram AI bubble strength: color-coded bars (blue/gray/red)")
print("   ✅ AECI-Var values: should show real numbers, not zeros")
print("   ✅ SECI values: should show real numbers in plots")
print("\n💡 If tipping points are still empty, that means no sharp transitions")
print("   were detected in your data (this can be normal for smooth changes).")
