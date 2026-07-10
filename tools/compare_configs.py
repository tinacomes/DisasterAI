#!/usr/bin/env python3
"""Compare primary-sweep results across model configurations on RAW metrics.

Built for the baseline-vs-switches comparison (immobile / components /
global-query baseline versus mobility=1 / spatial_bridged / network-query),
but works for any set of sweeps produced by test_filter_bubbles.py.

Usage:
    python3 tools/compare_configs.py <root> [save_dir]

<root> is a directory containing one experiment_results.json per
configuration, typically the download-artifact tree produced by
compare-network-mobility.yml:

    <root>/plots-config-baseline/experiment_results.json
    <root>/plots-config-switches/experiment_results.json

Outputs (into save_dir, default '.'):
    comparison_table.csv   per (config, alpha) raw metrics + SE — paper-ready numbers
    comparison_table.md    same table in Markdown, plus alpha* sensitivity per config
                           and paired per-seed deltas when exactly two configs share
                           replication counts
    comparison_configs.png overlay figure of the raw metrics vs alpha

IMPORTANT: configurations are compared on RAW metric values only. The
normalised composites (total_bubble_norm etc.) are range-normalised WITHIN
each sweep, so their values are not comparable across configurations; only
each configuration's own alpha* location is reported.

Paired deltas: run_replicated() seeds replicate i with seed i in every
configuration, so replicates are paired by seed and the delta CI uses the
per-seed differences (much tighter than the unpaired comparison).
"""
import sys
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure the repo root (parent of tools/) is importable regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_filter_bubbles import (
    load_results,
    compute_goldilocks_metrics,
    alpha_star_sensitivity,
    ALIGNMENT_SWEEP,
)

_SETTING_COLORS = ['#1A3A6B', '#B45F06', '#117733', '#882255', '#44AA99']

# (column header, metrics key for the mean, metrics key for the SE)
RAW_METRICS = [
    ('SECI',      'seci',     'seci_std'),
    ('AECI-Var',  'aeci',     'aeci_std'),
    ('AECI-Err',  'aeci_err', 'aeci_err_std'),
    ('MAE',       'mae',      'mae_std'),
    ('Unmet',     'unmet',    'unmet_std'),
    ('Precision', 'prec',     'prec_std'),
]

# metrics key of the per-replicate late-run lists (for paired deltas)
RUNS_KEYS = {
    'SECI':      'seci_runs',
    'AECI-Var':  'aeci_runs',
    'AECI-Err':  'aeci_err_runs',
    'MAE':       'mae_runs',
    'Unmet':     'unmet_runs',
    'Precision': 'prec_runs',
}


def setting_label(path):
    """.../plots-config-<label>/experiment_results.json -> '<label>'."""
    parent = os.path.basename(os.path.dirname(path))
    for prefix in ('plots-config-', 'plots-'):
        if parent.startswith(prefix):
            return parent[len(prefix):]
    return parent


def load_setting(path):
    all_results, *_ = load_results(path)
    metrics = compute_goldilocks_metrics(all_results)
    return setting_label(path), all_results, metrics


def write_csv(settings, save_dir):
    path = os.path.join(save_dir, 'comparison_table.csv')
    with open(path, 'w') as f:
        headers = ['config', 'alpha']
        for name, _, _ in RAW_METRICS:
            headers += [name, f'{name}_se']
        f.write(','.join(headers) + '\n')
        for label, _, metrics in settings:
            for a in (x for x in ALIGNMENT_SWEEP if x in metrics):
                row = [label, f'{a:.1f}']
                for _, mk, sk in RAW_METRICS:
                    row += [f'{metrics[a][mk]:.6g}', f'{metrics[a][sk]:.6g}']
                f.write(','.join(row) + '\n')
    print(f'Comparison CSV saved: {path}')
    return path


def paired_deltas(settings):
    """Per-alpha paired (by seed) deltas between exactly two settings.

    Returns {alpha: {metric: (mean_delta, ci95_halfwidth, n)}} for
    settings[1] − settings[0], or None when pairing is impossible.
    """
    if len(settings) != 2:
        return None
    (_, _, m0), (_, _, m1) = settings
    out = {}
    for a in (x for x in ALIGNMENT_SWEEP if x in m0 and x in m1):
        row = {}
        for name, runs_key in RUNS_KEYS.items():
            r0 = np.asarray(m0[a].get(runs_key, []), dtype=float)
            r1 = np.asarray(m1[a].get(runs_key, []), dtype=float)
            if len(r0) == 0 or len(r0) != len(r1):
                continue          # unequal replication counts → cannot pair
            d = r1 - r0           # replicate i has the same seed in both configs
            n = int(np.sum(~np.isnan(d)))
            if n < 2:
                continue
            mean = float(np.nanmean(d))
            se = float(np.nanstd(d, ddof=1) / np.sqrt(n))
            row[name] = (mean, 1.96 * se, n)
        if row:
            out[a] = row
    return out or None


def write_md(settings, save_dir):
    path = os.path.join(save_dir, 'comparison_table.md')
    with open(path, 'w') as f:
        f.write('# Configuration comparison — primary alignment sweep (raw metrics)\n\n')
        f.write('Late-run (last 75 ticks) means with across-replication standard '
                'errors. Raw values only: normalised composites are '
                'within-sweep quantities and must not be compared across '
                'configurations.\n\n')

        headers = ['config', 'α'] + [f'{n} (±SE)' for n, _, _ in RAW_METRICS]
        f.write('| ' + ' | '.join(headers) + ' |\n')
        f.write('|' + '---|' * len(headers) + '\n')
        for label, _, metrics in settings:
            for a in (x for x in ALIGNMENT_SWEEP if x in metrics):
                cells = [label, f'{a:.1f}']
                for _, mk, sk in RAW_METRICS:
                    cells.append(f'{metrics[a][mk]:.3f} (±{metrics[a][sk]:.3f})')
                f.write('| ' + ' | '.join(cells) + ' |\n')

        f.write('\n## α\\* sensitivity per configuration\n\n')
        f.write('α\\* is each configuration\'s own argmin (within-sweep '
                'normalisation); compare locations, not composite values.\n\n')
        f.write('| composite | ' + ' | '.join(label for label, _, _ in settings) + ' |\n')
        f.write('|---|' + '---|' * len(settings) + '\n')
        stars = {label: alpha_star_sensitivity(metrics)
                 for label, _, metrics in settings}
        for composite in next(iter(stars.values())):
            row = [composite.replace('|', '\\|')]
            row += [f'{stars[label][composite][0]}' for label, _, _ in settings]
            f.write('| ' + ' | '.join(row) + ' |\n')

        deltas = paired_deltas(settings)
        if deltas:
            l0, l1 = settings[0][0], settings[1][0]
            f.write(f'\n## Paired per-seed deltas: {l1} − {l0}\n\n')
            f.write('Replicate *i* uses seed *i* in every configuration, so '
                    'deltas are paired by seed; 95% CI = mean ± 1.96·SE of '
                    'the per-seed differences. A CI excluding 0 marks a '
                    'seed-robust configuration effect.\n\n')
            names = [n for n in RUNS_KEYS if any(n in d for d in deltas.values())]
            f.write('| α | ' + ' | '.join(f'Δ{n} [95% CI]' for n in names) + ' |\n')
            f.write('|---|' + '---|' * len(names) + '\n')
            for a in sorted(deltas):
                cells = [f'{a:.1f}']
                for n in names:
                    if n in deltas[a]:
                        mean, hw, _ = deltas[a][n]
                        mark = '**' if abs(mean) > hw else ''
                        cells.append(f'{mark}{mean:+.3f} [{mean - hw:+.3f}, {mean + hw:+.3f}]{mark}')
                    else:
                        cells.append('—')
                f.write('| ' + ' | '.join(cells) + ' |\n')
            f.write('\nBold = 95% CI excludes zero.\n')
        else:
            f.write('\n*Paired deltas skipped (needs exactly two configurations '
                    'with matching replication counts).*\n')
    print(f'Comparison Markdown saved: {path}')
    return path


def plot_comparison(settings, save_dir):
    """Overlay the RAW sweep metrics for every configuration."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Configuration comparison — primary alignment sweep (raw metrics)',
                 fontsize=14, fontweight='bold')

    panels = list(zip(axes.flat, RAW_METRICS))
    for i, (label, _, metrics) in enumerate(settings):
        c = _SETTING_COLORS[i % len(_SETTING_COLORS)]
        present = [a for a in ALIGNMENT_SWEEP if a in metrics]
        for ax, (name, mk, sk) in panels:
            ax.errorbar(present,
                        [metrics[a][mk] for a in present],
                        yerr=[metrics[a][sk] for a in present],
                        color=c, marker='o', markersize=5, linewidth=1.8,
                        capsize=3, alpha=0.9, label=label)

    titles = {
        'SECI':      'SECI vs α  (negative = social echo chamber)',
        'AECI-Var':  'AECI-Var vs α  (negative = AI echo chamber;\nL1+ beliefs, same pool as SECI)',
        'AECI-Err':  'AECI-Err vs α  (negative = AI-heavy agents\nmore confidently wrong)',
        'MAE':       'Belief MAE vs α  (disaster cells; lower = better)',
        'Unmet':     'Unmet needs vs α  (L3+ cells with zero relief per tick)',
        'Precision': 'Targeting precision vs α  (relief tokens on L3+ cells)',
    }
    for ax, (name, _, _) in panels:
        ax.set_title(titles[name])
        ax.set_xlabel('AI alignment level (α)')
        ax.set_ylabel(name)
        if name in ('SECI', 'AECI-Var', 'AECI-Err'):
            ax.axhline(0, color='k', ls=':', alpha=0.4)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, 'comparison_configs.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Comparison figure saved: {out}')


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else '.'
    save_dir = sys.argv[2] if len(sys.argv) > 2 else '.'
    paths = sorted(glob.glob(os.path.join(root, '**', 'experiment_results.json'),
                             recursive=True))
    if not paths:
        print(f'No experiment_results.json found under {root!r}')
        sys.exit(1)

    settings = [load_setting(p) for p in paths]
    print('Configurations found: ' + ', '.join(label for label, _, _ in settings))

    os.makedirs(save_dir, exist_ok=True)
    write_csv(settings, save_dir)
    write_md(settings, save_dir)
    plot_comparison(settings, save_dir)

    # Console echo of the Markdown table for the CI log
    with open(os.path.join(save_dir, 'comparison_table.md')) as f:
        print('\n' + f.read())


if __name__ == '__main__':
    main()
