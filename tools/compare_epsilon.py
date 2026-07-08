#!/usr/bin/env python3
"""Compare primary-sweep results across epsilon_decay settings.

Usage:
    python3 tools/compare_epsilon.py <root>

<root> is a directory that contains one experiment_results.json per setting,
typically the download-artifact tree produced by compare-epsilon-decay.yml:

    <root>/plots-epsilon-1.0/experiment_results.json
    <root>/plots-epsilon-0.98/experiment_results.json

For each setting it prints the Goldilocks alpha* (both the bubble-only and the
+MAE composite), the metric values at that alpha*, and the mode-choice
diagnostics (overall AI-query share and exploration_rate) averaged across the
sweep. This makes the effect of annealing epsilon directly visible: if the
sweep outcomes barely move while the exploration_rate collapses, the results
were being dominated by the exploration floor.
"""
import sys
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure the repo root (parent of tools/) is importable regardless of cwd,
# since running "python3 tools/compare_epsilon.py" puts tools/ on sys.path[0].
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_filter_bubbles import (
    load_results,
    compute_goldilocks_metrics,
    ALIGNMENT_SWEEP,
)

# Distinct, colourblind-safe colours per setting (extended if needed).
_SETTING_COLORS = ['#1A3A6B', '#B45F06', '#117733', '#882255', '#44AA99']


def setting_label(path):
    """.../plots-epsilon-<decay>/experiment_results.json -> 'decay=<decay>'."""
    parent = os.path.basename(os.path.dirname(path))
    return parent.replace('plots-epsilon-', 'decay=')


def _avg_over_sweep(all_results, key, *path):
    """Mean over alpha of all_results[i][key][path...], NaN-safe."""
    vals = []
    for r in all_results:
        d = r.get(key)
        for p in path:
            d = d.get(p) if isinstance(d, dict) else None
        if isinstance(d, (int, float)):
            vals.append(float(d))
    return float(np.mean(vals)) if vals else float('nan')


def load_setting(path):
    """Return (label, all_results, metrics) for one experiment_results.json."""
    all_results, *_ = load_results(path)
    metrics = compute_goldilocks_metrics(all_results)
    return setting_label(path), all_results, metrics


def summarise(all_results, metrics):
    alphas = [a for a in ALIGNMENT_SWEEP if a in metrics]
    bubble = {a: metrics[a]['total_bubble_norm'] for a in alphas}
    score = {a: metrics[a]['total_score_norm'] for a in alphas}
    astar = min(bubble, key=bubble.get)
    astar_score = min(score, key=score.get)
    return {
        'astar_bubble': astar,
        'astar_score': astar_score,
        'seci': metrics[astar]['seci'],
        'aeci': metrics[astar]['aeci'],
        'mae': metrics[astar]['mae'],
        'ai_share_ex': _avg_over_sweep(all_results, 'mode_choice_exploit', 'overall', 'ai'),
        'ai_share_er': _avg_over_sweep(all_results, 'mode_choice_explor', 'overall', 'ai'),
        'expl_rate_ex': _avg_over_sweep(all_results, 'mode_choice_exploit', 'exploration_rate'),
        'expl_rate_er': _avg_over_sweep(all_results, 'mode_choice_explor', 'exploration_rate'),
    }


def _series(all_results, key, *path):
    """Per-alpha series (aligned to ALIGNMENT_SWEEP) of a nested mode_choice value."""
    out = []
    for r in all_results:
        d = r.get(key)
        for p in path:
            d = d.get(p) if isinstance(d, dict) else None
        out.append(float(d) if isinstance(d, (int, float)) else float('nan'))
    return out


def plot_comparison(settings, save_dir):
    """Overlay the sweep for every setting on shared Goldilocks axes.

    `settings` is a list of (label, all_results, metrics). Saves
    comparison_epsilon.png with 6 panels vs alpha: the normalised bubble
    composite (with alpha* markers), SECI, AECI-Var, MAE, AI-query share, and
    the exploration rate (which visualises the floor collapsing under decay).
    """
    alphas = ALIGNMENT_SWEEP
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Epsilon-decay comparison — primary alignment sweep',
                 fontsize=14, fontweight='bold')

    def line(ax, xs, ys, color, label, ls='-', marker='o'):
        ax.plot(xs, ys, ls, color=color, marker=marker, markersize=5,
                linewidth=1.8, label=label, alpha=0.9)

    for i, (label, all_results, metrics) in enumerate(settings):
        c = _SETTING_COLORS[i % len(_SETTING_COLORS)]
        present = [a for a in alphas if a in metrics]
        comp = [metrics[a]['total_bubble_norm'] for a in present]
        astar = present[int(np.argmin(comp))]

        # (0,0) Goldilocks composite + alpha* marker
        line(axes[0, 0], present, comp, c, label)
        axes[0, 0].axvline(astar, color=c, ls='--', lw=1.5, alpha=0.6)

        line(axes[0, 1], present, [metrics[a]['seci'] for a in present], c, label)
        line(axes[0, 2], present, [metrics[a]['aeci'] for a in present], c, label)
        line(axes[1, 0], present, [metrics[a]['mae'] for a in present], c, label)

        # (1,1) AI-query share: exploit solid, explor dashed
        line(axes[1, 1], alphas, _series(all_results, 'mode_choice_exploit', 'overall', 'ai'),
             c, f'{label} (exploit)', ls='-', marker='o')
        line(axes[1, 1], alphas, _series(all_results, 'mode_choice_explor', 'overall', 'ai'),
             c, f'{label} (explor)', ls='--', marker='^')

        # (1,2) exploration rate (exploit; explor is ~identical)
        line(axes[1, 2], alphas, _series(all_results, 'mode_choice_exploit', 'exploration_rate'),
             c, label)

    axes[0, 0].set_title('Goldilocks composite  |SECI|n + |AECI-Var|n\n(dashed line = alpha*)')
    axes[0, 0].set_ylabel('Normalised composite (0 = best)')
    axes[0, 1].set_title('SECI vs alpha  (negative = social echo chamber)')
    axes[0, 1].set_ylabel('SECI'); axes[0, 1].axhline(0, color='k', ls=':', alpha=0.4)
    axes[0, 2].set_title('AECI-Var vs alpha  (negative = AI echo chamber)')
    axes[0, 2].set_ylabel('AECI-Var'); axes[0, 2].axhline(0, color='k', ls=':', alpha=0.4)
    axes[1, 0].set_title('Belief MAE vs alpha  (lower = more accurate)')
    axes[1, 0].set_ylabel('MAE (disaster cells)')
    axes[1, 1].set_title('AI-query share vs alpha\n(solid = exploitative, dashed = exploratory)')
    axes[1, 1].set_ylabel('Fraction of queries to AI')
    axes[1, 2].set_title('Exploration rate vs alpha\n(random source-choice fraction)')
    axes[1, 2].set_ylabel('exploration_rate')

    for ax in axes.flat:
        ax.set_xlabel('AI alignment level (alpha)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, 'comparison_epsilon.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nComparison figure saved: {out}')


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else '.'
    save_dir = sys.argv[2] if len(sys.argv) > 2 else '.'
    paths = sorted(glob.glob(os.path.join(root, '**', 'experiment_results.json'),
                             recursive=True))
    if not paths:
        print(f'No experiment_results.json found under {root!r}')
        sys.exit(1)

    settings = [load_setting(p) for p in paths]

    print('=' * 92)
    print('EPSILON-DECAY COMPARISON — primary alignment sweep')
    print('=' * 92)
    hdr = (f"{'setting':<16} {'a*(bub)':>8} {'a*(+MAE)':>9} {'SECI*':>7} "
           f"{'AECIv*':>7} {'MAE*':>7} {'AIsh_Ex':>8} {'AIsh_Er':>8} "
           f"{'expl_Ex':>8} {'expl_Er':>8}")
    print(hdr)
    print('-' * len(hdr))
    for label, all_results, metrics in settings:
        s = summarise(all_results, metrics)
        print(f"{label:<16} {s['astar_bubble']:>8} {s['astar_score']:>9} "
              f"{s['seci']:>7.3f} {s['aeci']:>7.3f} {s['mae']:>7.3f} "
              f"{s['ai_share_ex']:>8.3f} {s['ai_share_er']:>8.3f} "
              f"{s['expl_rate_ex']:>8.3f} {s['expl_rate_er']:>8.3f}")
    print()
    print('a*(bub)  = Goldilocks alpha, argmin |SECI|n + |AECI-Var|n')
    print('a*(+MAE) = argmin of the same composite plus the MAE penalty')
    print('SECI*/AECIv*/MAE* = metric values at a*(bub)')
    print('AIsh_Ex/Er = overall AI-query share (exploitative / exploratory), swept mean')
    print('expl_Ex/Er = exploration_rate (fraction of source choices that were random)')

    plot_comparison(settings, save_dir)


if __name__ == '__main__':
    main()
