"""
Goldilocks AI Alignment Experiment: Social vs. AI Filter Bubble Interplay

Research Question:
What is the optimal AI alignment level α* that breaks social echo chambers
without creating AI-enforced filter bubbles?

Background — Triple Social Bubbles:
1. Ego-network bubble: agents share/accept info primarily from friends
2. Agent-type cluster: exploiters cluster with exploiters, explorers with explorers
3. Shared-observation bubble: agents near same cells converge via direct sensing

AI Alignment Interplay:
- High alignment (AI confirms user beliefs): AI penetrates networks but amplifies all three bubbles
- Low alignment (AI reports truth): AI may be rejected (D/delta mechanism rejects divergent info)
- Goldilocks α*: AI is trusted and used, but divergent enough to disrupt bubble consensus

Metrics:
- SECI (Social Echo Chamber Index): -1 to +1
  * -1 = Strong echo chamber (friends very similar)
  *  0 = No echo chamber effect
  * +1 = Anti-echo chamber (friends more diverse)
- AECI (AI Echo Chamber Index): -1 to +1  (same variance formula as SECI)
  * -1 = AI-reliant agents have much lower belief variance than global (AI bubble)
  *  0 = No AI echo chamber effect
  * +1 = AI-reliant agents are more belief-diverse than global
- total_bubble = |SECI| + |AECI|  (minimise both)
- Belief MAE: accuracy cost at each alignment level

Goldilocks detection: argmin of total_bubble across alignment sweep
"""

import numpy as np
import matplotlib.pyplot as plt
from DisasterAI_Model import DisasterModel, HumanAgent
import os

base_params = {
    'share_exploitative': 0.5,
    'share_of_disaster': 0.15,
    'initial_trust': 0.3,
    'initial_ai_trust': 0.25,
    'number_of_humans': 100,
    'share_confirming': 0.7,
    'disaster_dynamics': 2,
    'width': 30,
    'height': 30,
    'ticks': 100,
    'learning_rate': 0.1,
    'epsilon': 0.3,
    'exploit_trust_lr': 0.015,
    'explor_trust_lr': 0.03,
}

ALIGNMENT_SWEEP = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # 6 levels; halved from 11 to cut runtime ~45%
STEADY_STATE_WINDOW = 15  # last N ticks for final metrics


def run_alignment_condition(ai_alignment, label):
    """Run one alignment condition and return per-tick metrics."""
    print(f"\n{'='*60}")
    print(f"Running: {label}  (alignment={ai_alignment})")
    print(f"{'='*60}")

    params = base_params.copy()
    params['ai_alignment_level'] = ai_alignment
    model = DisasterModel(**params)

    seci_exploit, seci_explor = [], []
    aeci_exploit, aeci_explor = [], []
    mae_exploit, mae_explor = [], []

    for tick in range(params['ticks']):
        model.step()

        if model.seci_data:
            s = model.seci_data[-1]
            seci_exploit.append(s[1])
            seci_explor.append(s[2])

        if model.aeci_data:
            a = model.aeci_data[-1]
            aeci_exploit.append(a[1])
            aeci_explor.append(a[2])

        if tick % 10 == 0:
            ex_errors, er_errors = [], []
            for agent in model.agent_list:
                if not isinstance(agent, HumanAgent):
                    continue
                err = np.mean([
                    abs(b.get('level', 0) - model.disaster_grid[c])
                    for c, b in agent.beliefs.items()
                    if isinstance(b, dict)
                ]) if agent.beliefs else 0
                (ex_errors if agent.agent_type == "exploitative" else er_errors).append(err)
            mae_exploit.append(np.mean(ex_errors) if ex_errors else 0)
            mae_explor.append(np.mean(er_errors) if er_errors else 0)

    return {
        'seci_exploit': seci_exploit,
        'seci_explor': seci_explor,
        'aeci_exploit': aeci_exploit,
        'aeci_explor': aeci_explor,
        'mae_exploit': mae_exploit,
        'mae_explor': mae_explor,
    }


def steady_state_mean(series, window=STEADY_STATE_WINDOW):
    """Mean of last `window` values."""
    if not series:
        return float('nan')
    return float(np.mean(series[-window:]))


def compute_goldilocks_metrics(all_results):
    """
    For each alignment level, compute steady-state SECI, AECI, and total_bubble.
    Both SECI and AECI use the same variance formula (-1 to +1).
    total_bubble = |SECI| + |AECI|  (minimise — both measure echo chamber intensity)
    """
    metrics = {}
    for alpha, res in zip(ALIGNMENT_SWEEP, all_results):
        seci_ss = (steady_state_mean(res['seci_exploit']) +
                   steady_state_mean(res['seci_explor'])) / 2
        aeci_ss = (steady_state_mean(res['aeci_exploit']) +
                   steady_state_mean(res['aeci_explor'])) / 2
        mae_ss = (steady_state_mean(res['mae_exploit']) +
                  steady_state_mean(res['mae_explor'])) / 2
        total_bubble = abs(seci_ss) + abs(aeci_ss)
        metrics[alpha] = {
            'seci': seci_ss,
            'aeci': aeci_ss,
            'mae': mae_ss,
            'total_bubble': total_bubble,
        }
    return metrics


def plot_goldilocks(metrics, all_results, save_dir):
    """Four-panel goldilocks summary figure."""
    alphas = ALIGNMENT_SWEEP
    seci_vals = [metrics[a]['seci'] for a in alphas]
    aeci_vals = [metrics[a]['aeci'] for a in alphas]
    total_vals = [metrics[a]['total_bubble'] for a in alphas]
    mae_vals = [metrics[a]['mae'] for a in alphas]

    best_alpha = alphas[int(np.argmin(total_vals))]
    print(f"\nGoldilocks α* = {best_alpha}  (minimum total_bubble = {min(total_vals):.3f})")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        'Goldilocks AI Alignment: Social vs. AI Filter Bubble Interplay\n'
        f'(α* = {best_alpha} minimises total bubble intensity)',
        fontsize=13, fontweight='bold'
    )

    # Panel 1: SECI vs alignment
    ax = axes[0, 0]
    ax.plot(alphas, seci_vals, 'b-o', linewidth=2, label='SECI (combined)')
    ax.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax.axvline(best_alpha, color='gold', linestyle='--', linewidth=2, label=f'α*={best_alpha}')
    ax.set_title('Social Echo Chamber Index vs Alignment\n(More negative = stronger social bubble)')
    ax.set_xlabel('AI Alignment Level (α)')
    ax.set_ylabel('SECI (-1 to +1)')
    ax.set_ylim(-1.1, 1.1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: AECI vs alignment
    ax = axes[0, 1]
    ax.plot(alphas, aeci_vals, 'r-o', linewidth=2, label='AECI (combined)')
    ax.axvline(best_alpha, color='gold', linestyle='--', linewidth=2, label=f'α*={best_alpha}')
    ax.set_title('AI Echo Chamber Index vs Alignment\n(More negative = stronger AI-induced bubble)')
    ax.set_xlabel('AI Alignment Level (α)')
    ax.set_ylabel('AECI (-1 to +1)')
    ax.set_ylim(-1.1, 1.1)
    ax.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Total bubble + goldilocks
    ax = axes[1, 0]
    ax.plot(alphas, total_vals, 'k-o', linewidth=2.5, label='total_bubble = |SECI| + |AECI|')
    ax.plot(best_alpha, min(total_vals), 'g*', markersize=18, zorder=5, label=f'Goldilocks α*={best_alpha}')
    ax.fill_between(alphas, total_vals, alpha=0.15, color='purple')
    ax.set_title('Total Bubble Intensity vs Alignment\n(Minimise to find goldilocks zone)')
    ax.set_xlabel('AI Alignment Level (α)')
    ax.set_ylabel('|SECI| + |AECI|')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 4: Belief MAE vs alignment (accuracy cost)
    ax = axes[1, 1]
    ax.plot(alphas, mae_vals, 'm-o', linewidth=2, label='Belief MAE (combined)')
    ax.axvline(best_alpha, color='gold', linestyle='--', linewidth=2, label=f'α*={best_alpha}')
    ax.set_title('Belief Accuracy vs Alignment\n(Lower MAE = more accurate beliefs)')
    ax.set_xlabel('AI Alignment Level (α)')
    ax.set_ylabel('Mean Absolute Error')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, 'goldilocks_alignment_sweep.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Goldilocks figure saved: {path}")

    # Also plot SECI/AECI time-series for a few key alignments
    _plot_timeseries(all_results, save_dir)
    return fig


def _plot_timeseries(all_results, save_dir):
    """Time-series SECI/AECI for selected alignment levels."""
    key_idxs = [0, 1, 2, 3, 4, 5]  # 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
    colors = plt.cm.viridis(np.linspace(0, 1, len(key_idxs)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('SECI and AECI Time-Series for Key Alignment Levels', fontsize=12)

    for color, idx in zip(colors, key_idxs):
        alpha = ALIGNMENT_SWEEP[idx]
        res = all_results[idx]
        label = f'α={alpha}'
        seci_comb = [(e + r) / 2 for e, r in zip(res['seci_exploit'], res['seci_explor'])]
        aeci_comb = [(e + r) / 2 for e, r in zip(res['aeci_exploit'], res['aeci_explor'])]
        ax1.plot(seci_comb, label=label, color=color, linewidth=1.8)
        ax2.plot(aeci_comb, label=label, color=color, linewidth=1.8)

    ax1.axhline(0, color='k', linestyle=':', alpha=0.4)
    ax1.set_title('SECI (Social Bubble) Over Time')
    ax1.set_xlabel('Tick')
    ax1.set_ylabel('SECI')
    ax1.set_ylim(-1.1, 1.1)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_title('AECI (AI Bubble) Over Time')
    ax2.set_xlabel('Tick')
    ax2.set_ylabel('AECI (-1 to +1)')
    ax2.axhline(0, color='k', linestyle=':', alpha=0.4)
    ax2.set_ylim(-1.1, 1.1)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'bubble_timeseries.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Time-series figure saved: {path}")


if __name__ == "__main__":
    print("=" * 70)
    print("GOLDILOCKS ALIGNMENT EXPERIMENT: SOCIAL vs. AI FILTER BUBBLE INTERPLAY")
    print("=" * 70)
    print(f"Sweeping alignment levels: {ALIGNMENT_SWEEP}")
    print(f"Ticks per run: {base_params['ticks']}")
    print(f"Steady-state window: last {STEADY_STATE_WINDOW} ticks\n")

    all_results = []
    for alpha in ALIGNMENT_SWEEP:
        res = run_alignment_condition(alpha, f"Alignment {alpha:.1f}")
        all_results.append(res)

    metrics = compute_goldilocks_metrics(all_results)

    print("\n" + "=" * 70)
    print("STEADY-STATE METRICS SUMMARY")
    print("=" * 70)
    print(f"{'α':>6}  {'SECI':>8}  {'AECI':>8}  {'|SECI|+AECI':>12}  {'MAE':>8}")
    print("-" * 50)
    for alpha in ALIGNMENT_SWEEP:
        m = metrics[alpha]
        marker = "  ← α*" if abs(m['total_bubble'] - min(v['total_bubble'] for v in metrics.values())) < 1e-9 else ""
        print(f"{alpha:>6.1f}  {m['seci']:>8.3f}  {m['aeci']:>8.3f}  {m['total_bubble']:>12.3f}  {m['mae']:>8.3f}{marker}")

    save_dir = 'test_results'  # relative path; works both locally and on CI
    plot_goldilocks(metrics, all_results, save_dir)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print("\nInterpretation guide:")
    print("  SECI < 0 : social echo chamber active (friends more similar than random)")
    print("  AECI < 0 : AI-induced bubble (AI-reliant agents more homogeneous than global)")
    print("  total_bubble = |SECI| + |AECI| : composite measure to minimise")
    print("  Goldilocks α* : minimises total_bubble")
    print("  Check MAE at α* : alignment gain should not sacrifice belief accuracy")
