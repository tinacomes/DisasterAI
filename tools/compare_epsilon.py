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

# Ensure the repo root (parent of tools/) is importable regardless of cwd,
# since running "python3 tools/compare_epsilon.py" puts tools/ on sys.path[0].
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_filter_bubbles import (
    load_results,
    compute_goldilocks_metrics,
    ALIGNMENT_SWEEP,
)


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


def summarise(path):
    all_results, *_ = load_results(path)
    metrics = compute_goldilocks_metrics(all_results)
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


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else '.'
    paths = sorted(glob.glob(os.path.join(root, '**', 'experiment_results.json'),
                             recursive=True))
    if not paths:
        print(f'No experiment_results.json found under {root!r}')
        sys.exit(1)

    print('=' * 92)
    print('EPSILON-DECAY COMPARISON — primary alignment sweep')
    print('=' * 92)
    hdr = (f"{'setting':<16} {'a*(bub)':>8} {'a*(+MAE)':>9} {'SECI*':>7} "
           f"{'AECIv*':>7} {'MAE*':>7} {'AIsh_Ex':>8} {'AIsh_Er':>8} "
           f"{'expl_Ex':>8} {'expl_Er':>8}")
    print(hdr)
    print('-' * len(hdr))
    for p in paths:
        s = summarise(p)
        print(f"{setting_label(p):<16} {s['astar_bubble']:>8} {s['astar_score']:>9} "
              f"{s['seci']:>7.3f} {s['aeci']:>7.3f} {s['mae']:>7.3f} "
              f"{s['ai_share_ex']:>8.3f} {s['ai_share_er']:>8.3f} "
              f"{s['expl_rate_ex']:>8.3f} {s['expl_rate_er']:>8.3f}")
    print()
    print('a*(bub)  = Goldilocks alpha, argmin |SECI|n + |AECI-Var|n')
    print('a*(+MAE) = argmin of the same composite plus the MAE penalty')
    print('SECI*/AECIv*/MAE* = metric values at a*(bub)')
    print('AIsh_Ex/Er = overall AI-query share (exploitative / exploratory), swept mean')
    print('expl_Ex/Er = exploration_rate (fraction of source choices that were random)')


if __name__ == '__main__':
    main()
