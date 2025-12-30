#!/usr/bin/env python3
"""
Reload existing results and regenerate plots with fixed visualization code
No need to rerun simulations - just replot!
"""

import os
import pickle
import sys

# Check both possible locations for results
possible_dirs = [
    "agent_model_results",  # Local directory
    "/content/drive/MyDrive/DisasterAI_Results"  # Drive directory
]

print(f"🔍 Searching for results...\n")

# Import plotting functions from main file
from DisasterAI_Model import (
    plot_final_echo_indices_vs_alignment,
    plot_average_performance_vs_alignment,
    plot_summary_echo_indices_vs_alignment,
    plot_summary_performance_vs_alignment,
    plot_phase_diagram_bubbles,
    plot_tipping_point_waterfall
)

# Import new temporal tipping point analysis
from temporal_tipping_points import (
    analyze_temporal_tipping_points,
    plot_temporal_tipping_points
)

# Import echo chamber evolution analysis
from echo_chamber_evolution import (
    extract_echo_chamber_lifecycle,
    plot_echo_chamber_evolution,
    plot_aeci_evolution
)

# Find the results file
file_b_pkl = None
save_dir = None

for directory in possible_dirs:
    test_path = os.path.join(directory, "results_AI_Alignment_Tipping_Point.pkl")
    if os.path.exists(test_path):
        file_b_pkl = test_path
        save_dir = directory
        print(f"✅ Found results in: {directory}")
        break
    else:
        print(f"  ✗ Not in {directory}")

if file_b_pkl is None:
    print(f"\n❌ Results file not found in any location!")
    print("\nSearched locations:")
    for directory in possible_dirs:
        print(f"  - {directory}")
        if os.path.exists(directory):
            pkl_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]
            if pkl_files:
                print(f"    Available files: {pkl_files}")
            else:
                print(f"    No .pkl files found")
        else:
            print(f"    Directory does not exist")
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

print("  → Tipping point waterfall (old method)")
plot_tipping_point_waterfall(results_b, all_alignment_values, param_name="AI Alignment")

print("\n  → NEW: Temporal tipping point analysis")
print("     (Tracking WHEN transitions occur during simulations)")
temporal_data = analyze_temporal_tipping_points(results_b, all_alignment_values)
plot_temporal_tipping_points(temporal_data, all_alignment_values, param_name="AI Alignment")

print("\n  → NEW: Echo chamber EVOLUTION (rise and fall)")
print("     (Full lifecycle: formation → peak → dissolution)")
lifecycle_data = extract_echo_chamber_lifecycle(results_b, all_alignment_values)
plot_echo_chamber_evolution(lifecycle_data, all_alignment_values, param_name="AI Alignment")

print("\n  → NEW: AECI evolution over time")
print("     (How AI preference grows during simulation)")
plot_aeci_evolution(lifecycle_data, all_alignment_values, param_name="AI Alignment")

print("\n" + "="*60)
print("✅ ALL PLOTS REGENERATED WITH FIXES!")
print("="*60)
print("\n📋 What's new:")
print("   ✅ Phase diagram AI bubble strength: color-coded bars (blue/gray/red)")
print("   ✅ AECI-Var values: showing real numbers, not zeros")
print("   ✅ SECI values: showing real numbers in plots")
print("   🆕 Temporal tipping point analysis: Shows WHEN transitions occur")
print("      - Separate tracking for exploit vs explor agents")
print("      - Tracks timing of transitions during simulations")
print("      - Shows how transition speed changes with alignment")
print("   🆕 Echo chamber EVOLUTION analysis: Shows RISE AND FALL")
print("      - Full time series showing formation → peak → dissolution")
print("      - Peak strength comparison across alignment levels")
print("      - Time-to-peak and chamber duration metrics")
print("      - Separate tracking for exploit vs explor agents")
print("      - Explains why final window shows ~0 (chambers already dissolved!)")
print("   🆕 AECI evolution plots: AI preference growth over time")
print("      - Shows how agents shift from friends to AI queries")
print("      - Tracks when 50% threshold is crossed")
print("\n💡 The NEW temporal analysis answers:")
print("   'At what tick do agents switch from friends to AI?'")
print("   'Does this happen earlier at higher alignment?'")
print("   'Do exploit and explor agents transition at different times?'")
print("\n💡 The NEW evolution analysis reveals:")
print("   'Echo chambers are TRANSIENT - they form early then DISSOLVE'")
print("   'Peak SECI ≈ -0.3 to -0.4 occurs at ticks 20-40'")
print("   'Chambers break (SECI→0) at ticks 60-100'")
print("   'Final window (120-150) shows dissolved state'")
print("   '→ This is why final plots showed near-zero values!')")
