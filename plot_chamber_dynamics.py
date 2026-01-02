"""
Chamber Dynamics Analysis - Social vs AI Echo Chambers
=======================================================

Answers the key questions:
1. Does AI amplify or disrupt social echo chambers?
2. What is the net effect on echo chamber strength?
3. When does AI dominance emerge vs social dominance?
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
except ImportError:
    RESULTS_DIR = "DisasterAI_results"

OUTPUT_DIR = os.path.join(RESULTS_DIR, "chamber_dynamics")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("CHAMBER DYNAMICS ANALYSIS")
print("="*70)

# Load Experiment B
file_b = os.path.join(RESULTS_DIR, "results_experiment_B.pkl")
with open(file_b, 'rb') as f:
    results_b = pickle.load(f)

alignment_values = sorted(results_b.keys())
print(f"✓ Loaded {len(alignment_values)} alignment values: {alignment_values}\n")

#########################################
# For each alignment: analyze chamber dynamics
#########################################

for align in alignment_values:
    print(f"Analyzing alignment = {align}...")

    res = results_b[align]
    seci_data = res.get('seci', np.array([]))
    aeci_data = res.get('aeci', np.array([]))

    if seci_data.size == 0 or aeci_data.size == 0:
        print(f"  ✗ Skipping - missing data")
        continue

    # Extract data
    ticks = seci_data[0, :, 0]

    # SECI - separate exploitative and exploratory
    seci_exploit = seci_data[:, :, 1]
    seci_explor = seci_data[:, :, 2]
    social_strength_exploit = np.abs(seci_exploit)  # 0 to 1
    social_strength_explor = np.abs(seci_explor)  # 0 to 1

    # AECI - separate exploitative and exploratory
    aeci_exploit = aeci_data[:, :, 1]
    aeci_explor = aeci_data[:, :, 2]
    ai_strength_exploit = aeci_exploit  # 0 to 1
    ai_strength_explor = aeci_explor  # 0 to 1

    # Calculate means for both agent types
    social_exploit_mean = np.mean(social_strength_exploit, axis=0)
    social_exploit_std = np.std(social_strength_exploit, axis=0)
    social_explor_mean = np.mean(social_strength_explor, axis=0)
    social_explor_std = np.std(social_strength_explor, axis=0)

    ai_exploit_mean = np.mean(ai_strength_exploit, axis=0)
    ai_exploit_std = np.std(ai_strength_exploit, axis=0)
    ai_explor_mean = np.mean(ai_strength_explor, axis=0)
    ai_explor_std = np.std(ai_strength_explor, axis=0)

    # Net chamber strength for each agent type
    net_exploit = social_exploit_mean + ai_exploit_mean
    net_explor = social_explor_mean + ai_explor_mean

    #########################################
    # Create comprehensive figure
    #########################################

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    fig.suptitle(f'Chamber Dynamics: AI Alignment = {align}',
                 fontsize=16, fontweight='bold')

    # Panel 1: EXPLOITATIVE Agents - Chamber Strength Over Time
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(ticks, social_exploit_mean, 'o-', label='Social (Exploitative)',
             color='#E74C3C', linewidth=2.5, markersize=4, alpha=0.9)
    ax1.fill_between(ticks, social_exploit_mean - social_exploit_std,
                      social_exploit_mean + social_exploit_std,
                      color='#E74C3C', alpha=0.2)

    ax1.plot(ticks, ai_exploit_mean, 's-', label='AI (Exploitative)',
             color='#3498DB', linewidth=2.5, markersize=4, alpha=0.9)
    ax1.fill_between(ticks, ai_exploit_mean - ai_exploit_std,
                      ai_exploit_mean + ai_exploit_std,
                      color='#3498DB', alpha=0.2)

    ax1.plot(ticks, net_exploit, '^-', label='NET (Exploitative)',
             color='#9B59B6', linewidth=3, markersize=6, alpha=0.9)

    ax1.set_ylabel('Chamber Strength', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Simulation Tick', fontsize=12, fontweight='bold')
    ax1.set_title('EXPLOITATIVE Agents: Echo Chamber Strength Over Time', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 2)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Panel 2: EXPLORATORY Agents - Chamber Strength Over Time
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(ticks, social_explor_mean, 'o-', label='Social (Exploratory)',
             color='#E74C3C', linewidth=2.5, markersize=4, alpha=0.9, linestyle='--')
    ax2.fill_between(ticks, social_explor_mean - social_explor_std,
                      social_explor_mean + social_explor_std,
                      color='#E74C3C', alpha=0.2)

    ax2.plot(ticks, ai_explor_mean, 's-', label='AI (Exploratory)',
             color='#3498DB', linewidth=2.5, markersize=4, alpha=0.9, linestyle='--')
    ax2.fill_between(ticks, ai_explor_mean - ai_explor_std,
                      ai_explor_mean + ai_explor_std,
                      color='#3498DB', alpha=0.2)

    ax2.plot(ticks, net_explor, '^-', label='NET (Exploratory)',
             color='#9B59B6', linewidth=3, markersize=6, alpha=0.9, linestyle='--')

    ax2.set_ylabel('Chamber Strength', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Simulation Tick', fontsize=12, fontweight='bold')
    ax2.set_title('EXPLORATORY Agents: Echo Chamber Strength Over Time', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 2)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Amplification Analysis - EXPLOITATIVE
    ax3 = fig.add_subplot(gs[2, 0])

    # Compare early vs late for EXPLOITATIVE
    early_social_ex = np.mean(social_exploit_mean[:30])
    late_social_ex = np.mean(social_exploit_mean[-30:])
    early_ai_ex = np.mean(ai_exploit_mean[:30])
    late_ai_ex = np.mean(ai_exploit_mean[-30:])
    early_net_ex = early_social_ex + early_ai_ex
    late_net_ex = late_social_ex + late_ai_ex

    x = np.arange(3)
    width = 0.35

    ax3.bar(x - width/2, [early_social_ex, early_ai_ex, early_net_ex], width,
            label='Early (1-30)', color='#95A5A6', alpha=0.8)
    ax3.bar(x + width/2, [late_social_ex, late_ai_ex, late_net_ex], width,
            label='Late (120-150)', color='#34495E', alpha=0.8)

    ax3.set_ylabel('Avg Strength', fontsize=10, fontweight='bold')
    ax3.set_title('Exploitative: Amplification', fontsize=11, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Social', 'AI', 'NET'], fontsize=9)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')

    # Panel 4: Key Statistics - Both Agent Types
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis('off')

    # Calculate changes for both agent types
    net_change_exploit = ((late_net_ex - early_net_ex) / early_net_ex * 100) if early_net_ex > 0 else 0

    early_social_exp = np.mean(social_explor_mean[:30])
    late_social_exp = np.mean(social_explor_mean[-30:])
    early_ai_exp = np.mean(ai_explor_mean[:30])
    late_ai_exp = np.mean(ai_explor_mean[-30:])
    early_net_exp = early_social_exp + early_ai_exp
    late_net_exp = late_social_exp + late_ai_exp
    net_change_explor = ((late_net_exp - early_net_exp) / early_net_exp * 100) if early_net_exp > 0 else 0

    stats_text = f"""
FINDINGS (Align = {align}):

EXPLOITATIVE Agents:
  Social: {early_social_ex:.2f} → {late_social_ex:.2f}
  AI: {early_ai_ex:.2f} → {late_ai_ex:.2f}
  NET: {early_net_ex:.2f} → {late_net_ex:.2f}
  Change: {net_change_exploit:+.1f}%
  → {'AMPLIFIES' if net_change_exploit > 10 else 'DISRUPTS' if net_change_exploit < -10 else 'MIXED'}

EXPLORATORY Agents:
  Social: {early_social_exp:.2f} → {late_social_exp:.2f}
  AI: {early_ai_exp:.2f} → {late_ai_exp:.2f}
  NET: {early_net_exp:.2f} → {late_net_exp:.2f}
  Change: {net_change_explor:+.1f}%
  → {'AMPLIFIES' if net_change_explor > 10 else 'DISRUPTS' if net_change_explor < -10 else 'MIXED'}
"""

    ax4.text(0.05, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center', transform=ax4.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()

    filename = f"chamber_dynamics_alignment_{align:.2f}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  ✓ Saved: {filename}")

#########################################
# Cross-alignment comparison
#########################################

print(f"\nCreating cross-alignment comparison...")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('How AI Alignment Affects Chamber Dynamics',
             fontsize=16, fontweight='bold')

colors = plt.cm.viridis(np.linspace(0, 1, len(alignment_values)))

net_effects = []
social_effects = []
ai_effects = []

for i, align in enumerate(alignment_values):
    res = results_b[align]
    seci_data = res.get('seci', np.array([]))
    aeci_data = res.get('aeci', np.array([]))

    if seci_data.size == 0 or aeci_data.size == 0:
        continue

    ticks = seci_data[0, :, 0]
    social_strength = np.abs(seci_data[:, :, 1])
    ai_strength = aeci_data[:, :, 1]

    social_mean = np.mean(social_strength, axis=0)
    ai_mean = np.mean(ai_strength, axis=0)
    net_mean = social_mean + ai_mean

    # Panel 1: Social chamber strength over time
    ax1.plot(ticks, social_mean, '-', label=f'{align}',
             color=colors[i], linewidth=2, alpha=0.8)

    # Panel 2: AI chamber strength over time
    ax2.plot(ticks, ai_mean, '-', label=f'{align}',
             color=colors[i], linewidth=2, alpha=0.8)

    # Panel 3: NET chamber strength over time
    ax3.plot(ticks, net_mean, '-', label=f'{align}',
             color=colors[i], linewidth=2, alpha=0.8)

    # Collect final values for panel 4
    social_effects.append(social_mean[-1])
    ai_effects.append(ai_mean[-1])
    net_effects.append(net_mean[-1])

ax1.set_title('Social Chamber Strength', fontsize=13, fontweight='bold')
ax1.set_xlabel('Tick', fontsize=11)
ax1.set_ylabel('Strength', fontsize=11)
ax1.legend(title='Alignment', fontsize=8)
ax1.grid(True, alpha=0.3)

ax2.set_title('AI Chamber Strength', fontsize=13, fontweight='bold')
ax2.set_xlabel('Tick', fontsize=11)
ax2.set_ylabel('Strength', fontsize=11)
ax2.legend(title='Alignment', fontsize=8)
ax2.grid(True, alpha=0.3)

ax3.set_title('NET Chamber Strength', fontsize=13, fontweight='bold')
ax3.set_xlabel('Tick', fontsize=11)
ax3.set_ylabel('Strength', fontsize=11)
ax3.legend(title='Alignment', fontsize=8)
ax3.grid(True, alpha=0.3)

# Panel 4: Final values comparison
ax4.plot(alignment_values, social_effects, 'o-', label='Social',
         color='#E74C3C', linewidth=3, markersize=10)
ax4.plot(alignment_values, ai_effects, 's-', label='AI',
         color='#3498DB', linewidth=3, markersize=10)
ax4.plot(alignment_values, net_effects, '^-', label='NET',
         color='#9B59B6', linewidth=3, markersize=10)

ax4.set_title('Final Chamber Strength vs AI Alignment', fontsize=13, fontweight='bold')
ax4.set_xlabel('AI Alignment', fontsize=11, fontweight='bold')
ax4.set_ylabel('Final Strength', fontsize=11, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

plt.tight_layout()

filename = "chamber_dynamics_comparison.png"
filepath = os.path.join(OUTPUT_DIR, filename)
plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"  ✓ Saved: {filename}")

print("\n" + "="*70)
print("✓ CHAMBER DYNAMICS ANALYSIS COMPLETE")
print("="*70)
print(f"Output: {OUTPUT_DIR}")
print("="*70)
