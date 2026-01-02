"""
Generate Plots for Experiment D - Q-Learning Parameter Robustness
==================================================================

Creates:
1. Heatmap showing SECI across learning rate × epsilon combinations
2. Individual plots for each parameter combination
3. Robustness variance analysis
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

OUTPUT_DIR = os.path.join(RESULTS_DIR, "experiment_d_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("EXPERIMENT D PLOTS - Q-LEARNING ROBUSTNESS")
print("="*70)
print(f"Loading from: {RESULTS_DIR}")
print(f"Saving to: {OUTPUT_DIR}")

# Load Experiment D results
file_d = os.path.join(RESULTS_DIR, "results_experiment_D.pkl")
if not os.path.exists(file_d):
    print(f"✗ ERROR: {file_d} not found")
    exit(1)

with open(file_d, 'rb') as f:
    results_d = pickle.load(f)

print(f"✓ Loaded Experiment D: {len(results_d)} parameter combinations")

# Extract parameter values
lr_values = sorted(set([k[0] for k in results_d.keys()]))
eps_values = sorted(set([k[1] for k in results_d.keys()]))

print(f"  Learning rates: {lr_values}")
print(f"  Epsilon values: {eps_values}\n")

#########################################
# Plot 1: SECI Heatmap
#########################################

print("Creating SECI heatmap...")

seci_matrix = np.zeros((len(lr_values), len(eps_values)))

for i, lr in enumerate(lr_values):
    for j, eps in enumerate(eps_values):
        key = (lr, eps)
        if key in results_d:
            res = results_d[key]
            seci_data = res.get('seci', np.array([]))
            if seci_data.size > 0 and seci_data.ndim >= 3:
                # Final SECI for exploitative agents
                final_seci = np.mean(seci_data[:, -1, 1])
                seci_matrix[i, j] = final_seci
            else:
                seci_matrix[i, j] = np.nan

fig, ax = plt.subplots(figsize=(10, 8))

# Create heatmap
im = ax.imshow(seci_matrix, cmap='RdYlGn_r', aspect='auto', vmin=-1, vmax=1)

# Set ticks
ax.set_xticks(range(len(eps_values)))
ax.set_yticks(range(len(lr_values)))
ax.set_xticklabels([f'{e:.1f}' for e in eps_values], fontsize=11)
ax.set_yticklabels([f'{lr:.2f}' for lr in lr_values], fontsize=11)

# Labels
ax.set_xlabel('Epsilon (Exploration Rate)', fontsize=13, fontweight='bold')
ax.set_ylabel('Learning Rate', fontsize=13, fontweight='bold')
ax.set_title('Final SECI Across Q-Learning Parameters\n(Robustness Check)',
             fontsize=15, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax, label='SECI (Exploitative Agents)')
cbar.ax.axhline(0, color='white', linestyle='--', linewidth=2)

# Add values to cells
for i in range(len(lr_values)):
    for j in range(len(eps_values)):
        val = seci_matrix[i, j]
        if not np.isnan(val):
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                   color=color, fontsize=10, fontweight='bold')

# Add variance info
variance = np.nanvar(seci_matrix)
ax.text(0.5, -0.15, f'Variance: {variance:.4f} (Low = Robust)',
        ha='center', transform=ax.transAxes, fontsize=12,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.tight_layout()
filepath = os.path.join(OUTPUT_DIR, "seci_heatmap.png")
plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"  ✓ Saved: seci_heatmap.png")

#########################################
# Plot 2: AECI Heatmap
#########################################

print("Creating AECI heatmap...")

aeci_matrix = np.zeros((len(lr_values), len(eps_values)))

for i, lr in enumerate(lr_values):
    for j, eps in enumerate(eps_values):
        key = (lr, eps)
        if key in results_d:
            res = results_d[key]
            aeci_data = res.get('aeci_variance', np.array([]))
            if isinstance(aeci_data, np.ndarray) and aeci_data.size > 0:
                if aeci_data.ndim == 3:
                    final_aeci = np.mean(aeci_data[:, -1, 1])
                elif aeci_data.ndim == 2:
                    final_aeci = np.mean(aeci_data[:, -1])
                else:
                    final_aeci = np.nan
                aeci_matrix[i, j] = final_aeci
            else:
                aeci_matrix[i, j] = np.nan

fig, ax = plt.subplots(figsize=(10, 8))

# Create heatmap
im = ax.imshow(aeci_matrix, cmap='viridis', aspect='auto')

# Set ticks
ax.set_xticks(range(len(eps_values)))
ax.set_yticks(range(len(lr_values)))
ax.set_xticklabels([f'{e:.1f}' for e in eps_values], fontsize=11)
ax.set_yticklabels([f'{lr:.2f}' for lr in lr_values], fontsize=11)

# Labels
ax.set_xlabel('Epsilon (Exploration Rate)', fontsize=13, fontweight='bold')
ax.set_ylabel('Learning Rate', fontsize=13, fontweight='bold')
ax.set_title('Final AECI-Var Across Q-Learning Parameters\n(Robustness Check)',
             fontsize=15, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax, label='AECI-Var')

# Add values to cells
for i in range(len(lr_values)):
    for j in range(len(eps_values)):
        val = aeci_matrix[i, j]
        if not np.isnan(val):
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                   color='white', fontsize=10, fontweight='bold')

# Add variance info
variance = np.nanvar(aeci_matrix)
ax.text(0.5, -0.15, f'Variance: {variance:.4f} (Low = Robust)',
        ha='center', transform=ax.transAxes, fontsize=12,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.tight_layout()
filepath = os.path.join(OUTPUT_DIR, "aeci_heatmap.png")
plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"  ✓ Saved: aeci_heatmap.png")

#########################################
# Plot 3: Robustness Summary
#########################################

print("Creating robustness summary...")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Experiment D: Q-Learning Parameter Robustness Analysis',
             fontsize=16, fontweight='bold')

# Panel 1: SECI variance across epsilon (for each LR)
for i, lr in enumerate(lr_values):
    seci_vals = []
    for eps in eps_values:
        key = (lr, eps)
        if key in results_d:
            res = results_d[key]
            seci_data = res.get('seci', np.array([]))
            if seci_data.size > 0 and seci_data.ndim >= 3:
                seci_vals.append(np.mean(seci_data[:, -1, 1]))
    ax1.plot(eps_values, seci_vals, 'o-', label=f'LR={lr:.2f}', linewidth=2, markersize=8)

ax1.set_xlabel('Epsilon', fontsize=12, fontweight='bold')
ax1.set_ylabel('Final SECI', fontsize=12, fontweight='bold')
ax1.set_title('SECI vs Epsilon (by Learning Rate)', fontsize=13, fontweight='bold')
ax1.axhline(0, color='black', linestyle='--', alpha=0.5)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Panel 2: SECI variance across LR (for each epsilon)
for j, eps in enumerate(eps_values):
    seci_vals = []
    for lr in lr_values:
        key = (lr, eps)
        if key in results_d:
            res = results_d[key]
            seci_data = res.get('seci', np.array([]))
            if seci_data.size > 0 and seci_data.ndim >= 3:
                seci_vals.append(np.mean(seci_data[:, -1, 1]))
    ax2.plot(lr_values, seci_vals, 's-', label=f'ε={eps:.1f}', linewidth=2, markersize=8)

ax2.set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
ax2.set_ylabel('Final SECI', fontsize=12, fontweight='bold')
ax2.set_title('SECI vs Learning Rate (by Epsilon)', fontsize=13, fontweight='bold')
ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Panel 3: Distribution of SECI values
all_seci = seci_matrix.flatten()
all_seci = all_seci[~np.isnan(all_seci)]
ax3.hist(all_seci, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
ax3.axvline(np.mean(all_seci), color='red', linestyle='--', linewidth=2, label=f'Mean={np.mean(all_seci):.3f}')
ax3.axvline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
ax3.set_xlabel('Final SECI', fontsize=12, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax3.set_title('Distribution of Final SECI Values', fontsize=13, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Panel 4: Variance statistics
stats_text = f"""
ROBUSTNESS STATISTICS:

SECI Variance: {np.var(all_seci):.6f}
SECI Std Dev: {np.std(all_seci):.4f}
SECI Range: [{np.min(all_seci):.4f}, {np.max(all_seci):.4f}]

Interpretation:
• Low variance = Robust to parameter changes
• High variance = Sensitive to parameters

Result: {'ROBUST' if np.var(all_seci) < 0.01 else 'MODERATE' if np.var(all_seci) < 0.05 else 'SENSITIVE'}
"""

ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
         verticalalignment='center', transform=ax4.transAxes,
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax4.axis('off')

plt.tight_layout()
filepath = os.path.join(OUTPUT_DIR, "robustness_summary.png")
plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"  ✓ Saved: robustness_summary.png")

print("\n" + "="*70)
print("✓ EXPERIMENT D PLOTS COMPLETE")
print("="*70)
print(f"Generated 3 plots in: {OUTPUT_DIR}")
print("  - seci_heatmap.png")
print("  - aeci_heatmap.png")
print("  - robustness_summary.png")
print("="*70)
