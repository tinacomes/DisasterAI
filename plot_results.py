#!/usr/bin/env python3
"""
Aggregate simulation JSON results (from simulate.py) and produce experiment plots.

Expects JSON files in --results-dir named by convention:
  alpha_<value>.json        — primary alignment sweep
  factor_rumor_<value>.json — rumour factor sweep
  factor_disaster_<value>.json — disaster tempo factor sweep
  factor_mix_<value>.json   — agent-type mix factor sweep

Produces three plots in --save-dir:
  goldilocks_alignment_sweep.png  — 2×3 scatter+errorbar, all metrics vs α
  bubble_timeseries.png           — 2×3 timeseries with ± std shading
  factor_comparison.png           — 3×3 bar chart, factor effects at α=0.5

Usage:
  python3 plot_results.py --results-dir results/ --save-dir test_results/
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Constants (must match simulate.py / test_filter_bubbles.py)
# ---------------------------------------------------------------------------
ALIGNMENT_SWEEP    = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
RUMOR_SWEEP        = [0.0, 0.5, 1.0]
DISASTER_SWEEP     = [0, 2, 3]
EXPLOITATIVE_SWEEP = [0.2, 0.5, 0.8]
STEADY_STATE_WINDOW = 15
FACTOR_ALPHA        = 0.5


# ---------------------------------------------------------------------------
# Data loading and aggregation
# ---------------------------------------------------------------------------

def load_and_aggregate(path):
    """Load a JSON result file and compute mean/std across runs."""
    with open(path) as f:
        data = json.load(f)

    runs = data['runs']
    keys = ['seci_exploit', 'seci_explor', 'aeci_exploit', 'aeci_explor',
            'mae_exploit',  'mae_explor',  'prec_exploit', 'prec_explor',
            'ai_query_ratio_exploit', 'ai_query_ratio_explor',
            'unmet_needs']

    result = {
        'condition': data['condition'],
        'n_runs':    data.get('n_runs', len(runs)),
        'metric_ticks': runs[0]['metric_ticks'],
        'n_ticks':   len(runs[0]['seci_exploit']),
    }
    for key in keys:
        arrays = []
        for run in runs:
            arrays.append([float('nan') if v is None else float(v) for v in run[key]])
        min_len = min(len(a) for a in arrays)
        mat = np.array([a[:min_len] for a in arrays], dtype=float)
        result[f'{key}_mean'] = np.nanmean(mat, axis=0).tolist()
        result[f'{key}_std']  = np.nanstd( mat, axis=0).tolist()
    return result


def load_results(results_dir):
    """Load all JSON files, return dicts keyed by condition value."""
    alpha_r    = {}  # float  -> aggregated result
    rumor_r    = {}  # float  -> aggregated result
    disaster_r = {}  # int    -> aggregated result
    mix_r      = {}  # float  -> aggregated result

    for path in sorted(glob.glob(os.path.join(results_dir, '*.json'))):
        fname = os.path.basename(path)
        try:
            res = load_and_aggregate(path)
        except Exception as e:
            print(f'  Warning: could not load {fname}: {e}')
            continue

        c = res['condition']
        if fname.startswith('alpha_'):
            alpha_r[float(c['ai_alignment_level'])] = res
        elif fname.startswith('factor_rumor_'):
            rumor_r[float(c['rumor_probability'])] = res
        elif fname.startswith('factor_disaster_'):
            disaster_r[int(c['disaster_dynamics'])] = res
        elif fname.startswith('factor_mix_'):
            mix_r[float(c['share_exploitative'])] = res
        else:
            print(f'  Skipping unrecognised file: {fname}')

    return alpha_r, rumor_r, disaster_r, mix_r


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def ss(arr, window=STEADY_STATE_WINDOW):
    """Steady-state mean: last `window` non-NaN values."""
    arr = [v for v in arr if not (isinstance(v, float) and np.isnan(v))]
    if not arr:
        return float('nan')
    return float(np.nanmean(arr[-window:]))


def ss_ms(res, key, window=STEADY_STATE_WINDOW):
    """(mean, std) of steady-state for a given metric key."""
    return ss(res[f'{key}_mean'], window), ss(res[f'{key}_std'], window)


def infer_n_runs(result_dict):
    if not result_dict:
        return None
    return next(iter(result_dict.values())).get('n_runs')


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _eb(ax, x, means, stds, color, fmt='-o', **kw):
    ax.errorbar(x, means, yerr=stds, fmt=fmt, color=color,
                linewidth=2, capsize=5, capthick=1.5, **kw)


def _band(ax, x, mean_arr, std_arr, color, label, lw=1.8):
    m = np.asarray(mean_arr)
    s = np.asarray(std_arr)
    x_ = list(x)[:len(m)]
    ax.plot(x_, m, color=color, linewidth=lw, label=label)
    ax.fill_between(x_, m - s, m + s, color=color, alpha=0.2)


def _finish(ax, title, xlabel, ylabel, ylim, hline=None, legend=True):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    lo, hi = ylim
    if lo is not None:
        ax.set_ylim(bottom=lo)
    if hi is not None:
        ax.set_ylim(top=hi)
    if hline is not None:
        ax.axhline(hline, color='k', linestyle=':', alpha=0.4)
    if legend:
        ax.legend(fontsize=8)


# ---------------------------------------------------------------------------
# goldilocks_alignment_sweep.png
# ---------------------------------------------------------------------------

def plot_goldilocks(alpha_r, save_dir):
    alphas = sorted(alpha_r)
    if not alphas:
        print('No alpha results — skipping goldilocks plot.')
        return

    n_runs = infer_n_runs(alpha_r)

    def ss_pair(key_e, key_r=None):
        key_r = key_r or key_e.replace('exploit', 'explor')
        return [
            ((ss(alpha_r[a][f'{key_e}_mean']) + ss(alpha_r[a][f'{key_r}_mean'])) / 2,
             (ss(alpha_r[a][f'{key_e}_std'])  + ss(alpha_r[a][f'{key_r}_std']))  / 2)
            for a in alphas
        ]

    seci_ms  = ss_pair('seci_exploit', 'seci_explor')
    aeci_ms  = ss_pair('aeci_exploit', 'aeci_explor')
    mae_ms   = ss_pair('mae_exploit',  'mae_explor')
    prec_ms  = ss_pair('prec_exploit', 'prec_explor')
    unmet_ms = [(ss(alpha_r[a]['unmet_needs_mean']),
                 ss(alpha_r[a]['unmet_needs_std'])) for a in alphas]

    total_vals = [abs(s) + abs(ae) for (s, _), (ae, _) in zip(seci_ms, aeci_ms)]
    best_alpha = alphas[int(np.nanargmin(total_vals))]
    n_label = f', N={n_runs} per level' if n_runs else ''

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f'Goldilocks AI Alignment (α*={best_alpha}) — mean ± std{n_label}',
        fontsize=13, fontweight='bold'
    )

    def ep(ax, ms_list, color, ylabel, title, ylim=(None, None)):
        means = [m for m, _ in ms_list]
        stds  = [s for _, s in ms_list]
        _eb(ax, alphas, means, stds, color)
        ax.axvline(best_alpha, color='gold', linestyle='--', linewidth=2,
                   label=f'α*={best_alpha}')
        _finish(ax, title, 'AI Alignment Level (α)', ylabel, ylim)

    ep(axes[0, 0], seci_ms, 'b', 'SECI (-1 to +1)',
       'Social Echo Chamber\n(negative = stronger bubble)', (-1.1, 1.1))
    axes[0, 0].axhline(0, color='k', linestyle=':', alpha=0.5)

    ep(axes[0, 1], aeci_ms, 'r', 'AECI (-1 to +1)',
       'AI-Induced Bubble\n(negative = stronger AI bubble)', (-1.1, 1.1))
    axes[0, 1].axhline(0, color='k', linestyle=':', alpha=0.5)

    ax = axes[0, 2]
    ax.plot(alphas, total_vals, 'k-o', linewidth=2.5, label='|SECI|+|AECI|')
    ax.plot(best_alpha, min(total_vals), 'g*', markersize=18, zorder=5,
            label=f'α*={best_alpha}')
    ax.fill_between(alphas, total_vals, alpha=0.15, color='purple')
    ax.axvline(best_alpha, color='gold', linestyle='--', linewidth=2)
    _finish(ax, 'Total Bubble Intensity\n(minimise)', 'α', '|SECI|+|AECI|', (0, None))

    ep(axes[1, 0], mae_ms, 'm', 'Mean Absolute Error',
       'Belief Accuracy\n(lower = beliefs closer to ground truth)')

    ep(axes[1, 1], unmet_ms, 'darkorange', 'Unmet high-need cells',
       'Unmet Needs (level ≥3, 0 tokens)\n(lower = better disaster response)')

    ep(axes[1, 2], prec_ms, 'teal', 'Correct / Total targets',
       'Relief Targeting Precision\n(higher = relief on high-need cells)', (0, 1.05))

    plt.tight_layout()
    path = os.path.join(save_dir, 'goldilocks_alignment_sweep.png')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')
    print(f'  Goldilocks α* = {best_alpha}  (total_bubble = {min(total_vals):.3f})')


# ---------------------------------------------------------------------------
# bubble_timeseries.png
# ---------------------------------------------------------------------------

def plot_timeseries(alpha_r, save_dir):
    alphas = sorted(alpha_r)
    if not alphas:
        print('No alpha results — skipping timeseries plot.')
        return

    n_runs = infer_n_runs(alpha_r)
    total_vals = [abs(ss(alpha_r[a]['seci_exploit_mean'])) +
                  abs(ss(alpha_r[a]['aeci_exploit_mean'])) for a in alphas]
    best_alpha = alphas[int(np.nanargmin(total_vals))]
    colors = plt.cm.viridis(np.linspace(0, 1, len(alphas)))
    n_label = f'  (mean ± std, N={n_runs})' if n_runs else ''

    # 3×3 layout: separate exploit/explor panels for SECI and AECI
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle(
        f'Filter Bubble & Delivery Metrics Over Time{n_label}',
        fontsize=13, fontweight='bold'
    )
    ax_seci_ex, ax_seci_er, ax_mae   = axes[0]
    ax_aeci_ex, ax_aeci_er, ax_unmet = axes[1]
    ax_prec,    ax_aqr_ex, ax_aqr_er  = axes[2]

    for color, alpha in zip(colors, alphas):
        res   = alpha_r[alpha]
        tf    = list(range(res['n_ticks']))
        ts    = res['metric_ticks']
        label = f'α={alpha}' + (' ★' if alpha == best_alpha else '')

        # SECI — separate exploit and explor panels
        _band(ax_seci_ex, tf, res['seci_exploit_mean'], res['seci_exploit_std'], color, label)
        _band(ax_seci_er, tf, res['seci_explor_mean'],  res['seci_explor_std'],  color, label)

        # AECI — separate exploit and explor panels
        _band(ax_aeci_ex, tf, res['aeci_exploit_mean'], res['aeci_exploit_std'], color, label)
        _band(ax_aeci_er, tf, res['aeci_explor_mean'],  res['aeci_explor_std'],  color, label)

        # MAE — combined, sampled
        mae_m = (np.array(res['mae_exploit_mean']) + np.array(res['mae_explor_mean'])) / 2
        mae_s = (np.array(res['mae_exploit_std'])  + np.array(res['mae_explor_std']))  / 2
        _band(ax_mae, ts[:len(mae_m)], mae_m, mae_s, color, label)

        # Precision — exploit (dashed) + explor (solid)
        for mk, ls, sfx in [('prec_exploit', '--', 'exploit'),
                             ('prec_explor',  '-',  'explor')]:
            pm = np.array(res[f'{mk}_mean'])
            ps = np.array(res[f'{mk}_std'])
            valid = ~np.isnan(pm)
            if np.any(valid):
                tv = np.array(ts[:len(pm)])[valid]
                ax_prec.plot(tv, pm[valid], color=color, linewidth=1.5,
                             linestyle=ls, label=f'α={alpha} {sfx}')
                ax_prec.fill_between(tv, (pm - ps)[valid], (pm + ps)[valid],
                                     color=color, alpha=0.15)

        # Unmet needs — per tick
        _band(ax_unmet, tf, res['unmet_needs_mean'], res['unmet_needs_std'],
              color, label)

        # AI query ratio — per tick (shows whether agents switch from AI to social)
        for mk, ax_d, ls in [
            ('ai_query_ratio_exploit', ax_aqr_ex, '-'),
            ('ai_query_ratio_explor',  ax_aqr_er, '-'),
        ]:
            if f'{mk}_mean' in res:
                qm = np.array(res[f'{mk}_mean'])
                qs = np.array(res[f'{mk}_std'])
                valid = ~np.isnan(qm)
                if np.any(valid):
                    tv = np.array(tf[:len(qm)])[valid]
                    ax_d.plot(tv, qm[valid], color=color, linewidth=1.5,
                              linestyle=ls, label=label)
                    ax_d.fill_between(tv, (qm - qs)[valid], (qm + qs)[valid],
                                      color=color, alpha=0.15)

    for ax, title, ylabel, ylim, hl in [
        (ax_seci_ex, 'SECI Over Time — Exploitative Agents\n(community variance vs global)',
         'SECI (-1 to +1)', (-1.1, 1.1), 0),
        (ax_seci_er, 'SECI Over Time — Exploratory Agents\n(community variance vs global)',
         'SECI (-1 to +1)', (-1.1, 1.1), 0),
        (ax_mae,     'Belief MAE Over Time  (exploit + explor avg, informed beliefs only)',
         'Mean Absolute Error', (0, None), None),
        (ax_aeci_ex, 'AECI Over Time — Exploitative Agents\n(AI-heavy vs AI-light within type)',
         'AECI (-1 to +1)', (-1.1, 1.1), 0),
        (ax_aeci_er, 'AECI Over Time — Exploratory Agents\n(AI-heavy vs AI-light within type)',
         'AECI (-1 to +1)', (-1.1, 1.1), 0),
        (ax_unmet,   'Unmet High-Need Cells per Tick (level ≥3, 0 tokens)',
         'Count', (0, None), None),
        (ax_prec,    'Relief Targeting Precision\n(solid=exploratory, dashed=exploitative)',
         'Correct / Total', (0, 1.05), 0.6),
        (ax_aqr_ex,  'AI Query Ratio — Exploitative Agents\n(fraction of queries sent to AI)',
         'AI / Total queries', (0, 1.05), None),
        (ax_aqr_er,  'AI Query Ratio — Exploratory Agents\n(↓ at high α = switch to social)',
         'AI / Total queries', (0, 1.05), None),
    ]:
        _finish(ax, title, 'Tick', ylabel, ylim, hline=hl)

    # Deduplicate precision legend
    hs, ls = ax_prec.get_legend_handles_labels()
    ax_prec.legend(hs[::2], [l.replace(' exploit', '') for l in ls[::2]], fontsize=8)

    plt.tight_layout()
    path = os.path.join(save_dir, 'bubble_timeseries.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ---------------------------------------------------------------------------
# factor_comparison.png
# ---------------------------------------------------------------------------

def plot_factor_comparison(rumor_r, disaster_r, mix_r, save_dir):
    n_runs = infer_n_runs(rumor_r) or infer_n_runs(disaster_r) or infer_n_runs(mix_r)
    n_label = f' (N={n_runs} per condition)' if n_runs else ''

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle(
        f'Factor Effects at α={FACTOR_ALPHA}{n_label}\n'
        'Each column: one factor swept; others held at base values',
        fontsize=12, fontweight='bold'
    )

    factor_cols = [
        ('Rumour probability\n(0=none  0.5=moderate  1=all communities)',
         RUMOR_SWEEP, rumor_r),
        ('Disaster tempo\n(0=static  2=medium  3=rapid)',
         DISASTER_SWEEP, disaster_r),
        ('Exploitative share\n(0.2=mostly exploratory  0.8=mostly exploitative)',
         EXPLOITATIVE_SWEEP, mix_r),
    ]
    row_metrics = [
        ('Steady-state SECI\n(negative = social bubble)',  'seci_exploit', (-1.1, 1.1)),
        ('Steady-state MAE\n(lower = accurate beliefs)',   'mae_exploit',  (0, None)),
        ('Final unmet needs\n(lower = better response)',   'unmet_needs',  (0, None)),
    ]
    bar_colors = ['#2196F3', '#FF9800', '#4CAF50']

    for col, (factor_label, factor_levels, res_dict) in enumerate(factor_cols):
        sorted_lvls = sorted(res_dict) if res_dict else factor_levels
        for row, (metric_label, metric_key, ylim) in enumerate(row_metrics):
            ax = axes[row, col]
            means, stds = [], []
            for lv in sorted_lvls:
                if lv not in res_dict:
                    means.append(float('nan'))
                    stds.append(0.0)
                    continue
                res = res_dict[lv]
                m = ss(res[f'{metric_key}_mean'])
                s = ss(res[f'{metric_key}_std'])
                means.append(m)
                stds.append(s if not np.isnan(s) else 0.0)

            x = np.arange(len(sorted_lvls))
            ax.bar(x, means, yerr=stds,
                   color=bar_colors[:len(sorted_lvls)],
                   alpha=0.8, capsize=6, edgecolor='white', linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels([str(v) for v in sorted_lvls], fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')

            if row == 0:
                ax.set_title(factor_label, fontsize=10, fontweight='bold')
            if col == 0:
                ax.set_ylabel(metric_label, fontsize=9)

            lo, hi = ylim
            if lo is not None:
                ax.set_ylim(bottom=lo)
            if hi is not None:
                ax.set_ylim(top=hi)
            if metric_key == 'seci_exploit':
                ax.axhline(0, color='k', linestyle=':', alpha=0.5)

    plt.tight_layout()
    path = os.path.join(save_dir, 'factor_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Aggregate simulate.py JSON outputs and produce experiment plots.'
    )
    parser.add_argument('--results-dir', default='results',
                        help='Directory containing JSON result files')
    parser.add_argument('--save-dir', default='test_results',
                        help='Directory to write PNG plots')
    args = parser.parse_args()

    print(f'Loading results from: {args.results_dir}')
    alpha_r, rumor_r, disaster_r, mix_r = load_results(args.results_dir)

    print(f'  Alpha levels found:   {sorted(alpha_r)}')
    print(f'  Rumour levels found:  {sorted(rumor_r)}')
    print(f'  Disaster levels:      {sorted(disaster_r)}')
    print(f'  Agent-mix levels:     {sorted(mix_r)}')

    os.makedirs(args.save_dir, exist_ok=True)

    if alpha_r:
        plot_goldilocks(alpha_r, args.save_dir)
        plot_timeseries(alpha_r, args.save_dir)
    else:
        print('Warning: no alpha_*.json files found — skipping primary plots.')

    if rumor_r or disaster_r or mix_r:
        plot_factor_comparison(rumor_r, disaster_r, mix_r, args.save_dir)
    else:
        print('Warning: no factor_*.json files found — skipping factor comparison plot.')

    print('Done.')


if __name__ == '__main__':
    main()
