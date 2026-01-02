"""
Generate Temporal Evolution Plots for Experiment B
===================================================

Creates clean, publication-quality plots showing how SECI and AECI
evolve over time for each AI alignment level.

This provides the temporal analysis that shows:
- How social echo chambers form and dissolve
- How AI echo chambers emerge
- The transition dynamics between social → AI dominance
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Configuration
try:
    from google.colab import drive
    if not os.path.exists('/content/drive/MyDrive'):
        drive.mount('/content/drive')
    RESULTS_DIR = "/content/drive/MyDrive/DisasterAI_results"
    IN_COLAB = True
except ImportError:
    RESULTS_DIR = "DisasterAI_results"
    IN_COLAB = False

OUTPUT_DIR = os.path.join(RESULTS_DIR, "temporal_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("TEMPORAL EVOLUTION PLOTS - EXPERIMENT B")
print("="*70)
print(f"Loading from: {RESULTS_DIR}")
print(f"Saving to: {OUTPUT_DIR}")

# Load Experiment B results
file_b = os.path.join(RESULTS_DIR, "results_experiment_B.pkl")
if not os.path.exists(file_b):
    print(f"✗ ERROR: {file_b} not found")
    exit(1)

with open(file_b, 'rb') as f:
    results_b = pickle.load(f)

alignment_values = sorted(results_b.keys())
print(f"✓ Loaded Experiment B: {len(alignment_values)} alignment values")
print(f"  Alignment values: {alignment_values}\n")

#########################################
# Plot 1: Individual temporal plots for each alignment
#########################################

for align in alignment_values:
    print(f"Creating temporal plot for alignment = {align}...")

    res = results_b[align]
    seci_data = res.get('seci', np.array([]))
    aeci_var_data = res.get('aeci_variance', np.array([]))

    if seci_data.size == 0 or seci_data.ndim < 3:
        print(f"  ✗ Skipping (missing SECI data)")
        continue

    # Extract data
    ticks = seci_data[0, :, 0]  # Tick numbers

    # SECI: Column 1 = exploitative, Column 2 = exploratory
    seci_exploit = seci_data[:, :, 1]  # All runs, all ticks, exploitative
    seci_explor = seci_data[:, :, 2]   # All runs, all ticks, exploratory

    # Calculate mean and std
    seci_exploit_mean = np.mean(seci_exploit, axis=0)
    seci_exploit_std = np.std(seci_exploit, axis=0)
    seci_explor_mean = np.mean(seci_explor, axis=0)
    seci_explor_std = np.std(seci_explor, axis=0)

    # AECI variance
    if isinstance(aeci_var_data, np.ndarray) and aeci_var_data.size > 0:
        if aeci_var_data.ndim == 3:
            aeci_mean = np.mean(aeci_var_data[:, :, 1], axis=0)
            aeci_std = np.std(aeci_var_data[:, :, 1], axis=0)
            has_aeci = True
        elif aeci_var_data.ndim == 2:
            aeci_mean = np.mean(aeci_var_data, axis=0)
            aeci_std = np.std(aeci_var_data, axis=0)
            has_aeci = True
        else:
            has_aeci = False
    else:
        has_aeci = False

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Panel 1: SECI (Social Echo Chambers)
    ax1.plot(ticks, seci_exploit_mean, 'o-', label='Exploitative',
             color='#E74C3C', linewidth=2.5, markersize=4, alpha=0.9)
    ax1.fill_between(ticks,
                      seci_exploit_mean - seci_exploit_std,
                      seci_exploit_mean + seci_exploit_std,
                      color='#E74C3C', alpha=0.2)

    ax1.plot(ticks, seci_explor_mean, 's-', label='Exploratory',
             color='#3498DB', linewidth=2.5, markersize=4, alpha=0.9)
    ax1.fill_between(ticks,
                      seci_explor_mean - seci_explor_std,
                      seci_explor_mean + seci_explor_std,
                      color='#3498DB', alpha=0.2)

    ax1.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7,
                label='No Chamber (0)')
    ax1.axhspan(-1, -0.05, alpha=0.05, color='red', label='Echo Chamber')
    ax1.axhspan(0.05, 1, alpha=0.05, color='green', label='Anti-Chamber')

    ax1.set_ylabel('SECI (Social Echo Chamber Index)', fontsize=13, fontweight='bold')
    ax1.set_ylim(-1, 1)
    ax1.set_title(f'AI Alignment = {align}', fontsize=15, fontweight='bold')
    ax1.legend(loc='best', fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle=':')

    # Panel 2: AECI (AI Echo Chambers)
    if has_aeci:
        ax2.plot(ticks, aeci_mean, 'o-', label='AECI-Var (AI Echo Chambers)',
                 color='#9B59B6', linewidth=2.5, markersize=4, alpha=0.9)
        ax2.fill_between(ticks,
                          aeci_mean - aeci_std,
                          aeci_mean + aeci_std,
                          color='#9B59B6', alpha=0.2)
        ax2.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        ax2.set_ylabel('AECI-Var (AI Echo Chamber Index)', fontsize=13, fontweight='bold')
        ax2.legend(loc='best', fontsize=11, framealpha=0.9)
        ax2.grid(True, alpha=0.3, linestyle=':')
    else:
        ax2.text(0.5, 0.5, 'AECI data not available',
                ha='center', va='center', transform=ax2.transAxes,
                fontsize=14, color='gray')

    ax2.set_xlabel('Simulation Tick', fontsize=13, fontweight='bold')

    plt.tight_layout()

    # Save
    filename = f"temporal_evolution_alignment_{align:.2f}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  ✓ Saved: {filename}")

#########################################
# Plot 2: Combined comparison (all alignments on one plot)
#########################################

print(f"\nCreating combined comparison plot...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle('Temporal Evolution Across AI Alignment Levels',
             fontsize=16, fontweight='bold')

colors = plt.cm.viridis(np.linspace(0, 1, len(alignment_values)))

for i, align in enumerate(alignment_values):
    res = results_b[align]
    seci_data = res.get('seci', np.array([]))
    aeci_var_data = res.get('aeci_variance', np.array([]))

    if seci_data.size == 0 or seci_data.ndim < 3:
        continue

    ticks = seci_data[0, :, 0]
    seci_exploit_mean = np.mean(seci_data[:, :, 1], axis=0)

    ax1.plot(ticks, seci_exploit_mean, '-',
             label=f'Align={align}', color=colors[i], linewidth=2, alpha=0.8)

    if isinstance(aeci_var_data, np.ndarray) and aeci_var_data.size > 0:
        if aeci_var_data.ndim == 3:
            aeci_mean = np.mean(aeci_var_data[:, :, 1], axis=0)
        elif aeci_var_data.ndim == 2:
            aeci_mean = np.mean(aeci_var_data, axis=0)
        else:
            continue
        ax2.plot(ticks, aeci_mean, '-',
                 label=f'Align={align}', color=colors[i], linewidth=2, alpha=0.8)

# SECI panel
ax1.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
ax1.set_xlabel('Simulation Tick', fontsize=12, fontweight='bold')
ax1.set_ylabel('SECI (Exploitative Agents)', fontsize=12, fontweight='bold')
ax1.set_title('Social Echo Chambers', fontsize=14, fontweight='bold')
ax1.set_ylim(-1, 1)
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)

# AECI panel
ax2.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
ax2.set_xlabel('Simulation Tick', fontsize=12, fontweight='bold')
ax2.set_ylabel('AECI-Var', fontsize=12, fontweight='bold')
ax2.set_title('AI Echo Chambers', fontsize=14, fontweight='bold')
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()

filename = "temporal_evolution_combined.png"
filepath = os.path.join(OUTPUT_DIR, filename)
plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"  ✓ Saved: {filename}")

print("\n" + "="*70)
print("✓ TEMPORAL PLOTS COMPLETE")
print("="*70)
print(f"Generated {len(alignment_values) + 1} plots")
print(f"Location: {OUTPUT_DIR}")
print("="*70)
