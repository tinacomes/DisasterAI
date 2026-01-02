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

    # SECI (exploitative agents) - negative = echo chamber
    # Take absolute value to get chamber STRENGTH
    seci_exploit = seci_data[:, :, 1]
    social_strength = np.abs(seci_exploit)  # 0 to 1, higher = stronger chamber

    # AECI (exploitative agents) - already 0 to 1
    aeci_exploit = aeci_data[:, :, 1]
    ai_strength = aeci_exploit  # 0 to 1

    # Calculate means
    social_mean = np.mean(social_strength, axis=0)
    social_std = np.std(social_strength, axis=0)
    ai_mean = np.mean(ai_strength, axis=0)
    ai_std = np.std(ai_strength, axis=0)

    # Net chamber strength: How strong are echo chambers overall?
    net_strength = social_mean + ai_mean

    # Dominance: Which type of chamber dominates?
    social_dominance = social_mean / (social_mean + ai_mean + 1e-10)  # 0 to 1
    ai_dominance = ai_mean / (social_mean + ai_mean + 1e-10)  # 0 to 1

    #########################################
    # Create comprehensive figure
    #########################################

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    fig.suptitle(f'Chamber Dynamics: AI Alignment = {align}',
                 fontsize=16, fontweight='bold')

    # Panel 1: Chamber Strength Over Time
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(ticks, social_mean, 'o-', label='Social Chamber Strength',
             color='#E74C3C', linewidth=2.5, markersize=4, alpha=0.9)
    ax1.fill_between(ticks, social_mean - social_std, social_mean + social_std,
                      color='#E74C3C', alpha=0.2)

    ax1.plot(ticks, ai_mean, 's-', label='AI Chamber Strength',
             color='#3498DB', linewidth=2.5, markersize=4, alpha=0.9)
    ax1.fill_between(ticks, ai_mean - ai_std, ai_mean + ai_std,
                      color='#3498DB', alpha=0.2)

    ax1.plot(ticks, net_strength, '^-', label='NET Chamber Strength (Social + AI)',
             color='#9B59B6', linewidth=3, markersize=6, alpha=0.9)

    ax1.set_ylabel('Chamber Strength', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Simulation Tick', fontsize=12, fontweight='bold')
    ax1.set_title('Echo Chamber Strength Over Time', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 2)  # Both can add up
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Dominance Over Time (Stacked Area)
    ax2 = fig.add_subplot(gs[1, :])
    ax2.fill_between(ticks, 0, social_dominance,
                      color='#E74C3C', alpha=0.6, label='Social Dominated')
    ax2.fill_between(ticks, social_dominance, 1,
                      color='#3498DB', alpha=0.6, label='AI Dominated')
    ax2.axhline(0.5, color='black', linestyle='--', linewidth=2, alpha=0.7,
                label='Equal Dominance')

    ax2.set_ylabel('Dominance', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Simulation Tick', fontsize=12, fontweight='bold')
    ax2.set_title('Social vs AI Dominance Over Time', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Amplification Analysis
    ax3 = fig.add_subplot(gs[2, 0])

    # Compare early vs late
    early_social = np.mean(social_mean[:30])
    late_social = np.mean(social_mean[-30:])
    early_ai = np.mean(ai_mean[:30])
    late_ai = np.mean(ai_mean[-30:])
    early_net = early_social + early_ai
    late_net = late_social + late_ai

    x = np.arange(3)
    width = 0.35

    ax3.bar(x - width/2, [early_social, early_ai, early_net], width,
            label='Early (ticks 1-30)', color='#95A5A6', alpha=0.8)
    ax3.bar(x + width/2, [late_social, late_ai, late_net], width,
            label='Late (ticks 120-150)', color='#34495E', alpha=0.8)

    ax3.set_ylabel('Average Chamber Strength', fontsize=11, fontweight='bold')
    ax3.set_title('Amplification Analysis', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Social', 'AI', 'NET'])
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    # Panel 4: Key Statistics
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis('off')

    # Find when AI overtakes social (if it does)
    overtake_tick = None
    for i, tick in enumerate(ticks):
        if ai_mean[i] > social_mean[i]:
            overtake_tick = int(tick)
            break

    # Calculate effects
    social_change = ((late_social - early_social) / early_social * 100) if early_social > 0 else 0
    ai_emergence = late_ai
    net_change = ((late_net - early_net) / early_net * 100) if early_net > 0 else 0

    stats_text = f"""
KEY FINDINGS (Alignment = {align}):

Social Chamber Change:
  Early: {early_social:.3f}
  Late:  {late_social:.3f}
  Change: {social_change:+.1f}%

AI Chamber Emergence:
  Early: {early_ai:.3f}
  Late:  {late_ai:.3f}
  Strength: {ai_emergence:.3f}

NET Effect:
  Early: {early_net:.3f}
  Late:  {late_net:.3f}
  Change: {net_change:+.1f}%

AI Overtakes Social:
  {'Tick ' + str(overtake_tick) if overtake_tick else 'Never'}

INTERPRETATION:
  {'AI AMPLIFIES chambers' if net_change > 10 else 'AI DISRUPTS chambers' if net_change < -10 else 'AI has MIXED effect'}
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
