"""Co-evolution analysis: how the social echo chamber (SECI) and the
AI-channel information environment (AECI-IE family) evolve together, and how
that couples to AI reliance.

Motivating hypothesis (author): at low α, exploratory agents are AI-reliant
and a shared-source dynamic could narrow them; exploitative agents start in a
social bubble and, as α rises, hand it over to an AI-induced bubble (the AI
stops correcting and starts mirroring, while their reliance on it grows).

Inputs: an experiment_results.json written by test_filter_bubbles.py (the
same file tools/compare_configs.py consumes). Outputs, per configuration:

  coevolution_timeseries.png — per-α rows × per-type columns: SECI(t),
      AECI-IE(t), AECI-IE-rel(t) (left axis, [−1,1]) and AI reliance(t)
      (right axis, [0,1]).
  coevolution_phase.png — SECI vs AECI-IE phase plane, one mean trajectory
      per α per type, time direction marked (○ start, ● end).
  coevolution_summary.csv / .md — late-run means per α and type: SECI,
      AECI-IE, AECI-IE-chan, AECI-IE-rel, SECI-IE, reliance, effective α.

Reading the phase plane against the hypothesis:
  social → AI handover (exploiters) = trajectory endpoint moves RIGHT on
      SECI (social chamber weakens) while reliance rises and AECI-IE-rel
      falls toward 0 (AI stops broadening their diet and mirrors it).
  shared-source narrowing (explorers) = endpoint moves DOWN on AECI-IE
      with α while reliance stays high.

Usage:
    python tools/coevolution.py <experiment_results.json> [--out DIR]
        [--alphas 0.0,0.5,1.0]
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_filter_bubbles import ALIGNMENT_SWEEP, ss   # noqa: E402

TYPES = (('exploit', 'Exploitative'), ('explor', 'Exploratory'))

# (label, series key prefix) — per-type series are f'{prefix}_{type}_mean'
SERIES = [
    ('SECI',         'seci'),
    ('AECI-IE',      'aeci_ie'),
    ('AECI-IE-rel',  'aeci_ie_rel'),
]
RELIANCE = 'ai_reliance'

SUMMARY_KEYS = ['seci', 'aeci_ie', 'aeci_ie_chan', 'aeci_ie_rel',
                'seci_ie', 'ai_reliance', 'effective_alpha']


def _series(res, key):
    v = res.get(key, [])
    return np.asarray(v, dtype=float) if v else np.array([])


def load(path):
    with open(path) as f:
        data = json.load(f)
    all_results = data['all_results']
    alphas = ALIGNMENT_SWEEP[:len(all_results)]
    return dict(zip(alphas, all_results))


def pick_alphas(per_alpha, requested):
    if requested:
        want = [float(x) for x in requested.split(',')]
        return [a for a in want if a in per_alpha]
    present = sorted(per_alpha)
    # default: endpoints + midpoint-ish
    picks = [present[0], present[len(present) // 2], present[-1]]
    return sorted(set(picks))


def plot_timeseries(per_alpha, alphas, out):
    fig, axes = plt.subplots(len(alphas), 2,
                             figsize=(14, 3.4 * len(alphas)),
                             squeeze=False, sharex=True)
    colors = {'SECI': 'tab:blue', 'AECI-IE': 'tab:red',
              'AECI-IE-rel': 'tab:orange'}
    for r, alpha in enumerate(alphas):
        res = per_alpha[alpha]
        for c, (tkey, tname) in enumerate(TYPES):
            ax = axes[r][c]
            for label, prefix in SERIES:
                y = _series(res, f'{prefix}_{tkey}_mean')
                if y.size:
                    ax.plot(np.arange(y.size) * 5, y, label=label,
                            color=colors[label], lw=1.6)
            ax.axhline(0, color='k', ls=':', alpha=0.4)
            ax.set_ylim(-1.05, 1.05)
            ax.set_ylabel('index [−1, 1]')
            ax.set_title(f'α={alpha:g} — {tname}')
            ax2 = ax.twinx()
            w = _series(res, f'{RELIANCE}_{tkey}_mean')
            if w.size:
                ax2.plot(np.arange(w.size) * 5, w, label='AI reliance',
                         color='tab:green', lw=1.4, ls='--')
            ax2.set_ylim(0, 1)
            ax2.set_ylabel('AI reliance share', color='tab:green')
            if r == 0 and c == 0:
                lines, labels = ax.get_legend_handles_labels()
                l2, lb2 = ax2.get_legend_handles_labels()
                ax.legend(lines + l2, labels + lb2, loc='lower left',
                          fontsize=8, ncol=2)
    for c in range(2):
        axes[-1][c].set_xlabel('tick')
    fig.suptitle('Co-evolution of social (SECI) and AI-channel (AECI-IE) '
                 'indices with AI reliance', y=1.0)
    fig.tight_layout()
    path = os.path.join(out, 'coevolution_timeseries.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')


def plot_phase(per_alpha, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    alphas = sorted(per_alpha)
    cmap = plt.get_cmap('viridis')
    for c, (tkey, tname) in enumerate(TYPES):
        ax = axes[c]
        for i, alpha in enumerate(alphas):
            res = per_alpha[alpha]
            x = _series(res, f'seci_{tkey}_mean')
            y = _series(res, f'aeci_ie_{tkey}_mean')
            n = min(x.size, y.size)
            if n < 2:
                continue
            x, y = x[:n], y[:n]
            color = cmap(i / max(1, len(alphas) - 1))
            ax.plot(x, y, color=color, lw=1.3, alpha=0.85,
                    label=f'α={alpha:g}')
            ax.plot(x[0], y[0], marker='o', mfc='none', color=color, ms=7)
            ax.plot(x[-1], y[-1], marker='o', color=color, ms=7)
        ax.axhline(0, color='k', ls=':', alpha=0.4)
        ax.axvline(0, color='k', ls=':', alpha=0.4)
        ax.set_xlabel('SECI (social echo chamber; negative = chamber)')
        ax.set_ylabel('AECI-IE (AI-channel narrowing; negative = narrow)')
        ax.set_title(f'{tname} — ○ start, ● steady state')
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        if c == 0:
            ax.legend(fontsize=8, loc='upper left')
    fig.suptitle('Phase plane: social vs AI-channel bubble per α')
    fig.tight_layout()
    path = os.path.join(out, 'coevolution_phase.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')


def write_summary(per_alpha, out):
    rows = []
    for alpha in sorted(per_alpha):
        res = per_alpha[alpha]
        for tkey, tname in TYPES:
            row = {'alpha': alpha, 'type': tkey}
            for k in SUMMARY_KEYS:
                row[k] = ss(res.get(f'{k}_{tkey}_mean', []))
            rows.append(row)
    csv_path = os.path.join(out, 'coevolution_summary.csv')
    with open(csv_path, 'w') as f:
        headers = ['alpha', 'type'] + SUMMARY_KEYS
        f.write(','.join(headers) + '\n')
        for row in rows:
            f.write(','.join(f'{row[h]:.4g}' if isinstance(row[h], float)
                             else str(row[h]) for h in headers) + '\n')
    md_path = os.path.join(out, 'coevolution_summary.md')
    with open(md_path, 'w') as f:
        f.write('# Co-evolution summary (late-run means)\n\n')
        f.write('AECI-IE-rel: + = channel broadens the community\'s own '
                'belief diversity, − = narrows it, ≈0 = mirrors it.\n\n')
        f.write('| ' + ' | '.join(['alpha', 'type'] + SUMMARY_KEYS) + ' |\n')
        f.write('|' + '---|' * (2 + len(SUMMARY_KEYS)) + '\n')
        for row in rows:
            cells = [f'{row["alpha"]:g}', row['type']] + [
                f'{row[k]:.3f}' if np.isfinite(row[k]) else '—'
                for k in SUMMARY_KEYS]
            f.write('| ' + ' | '.join(cells) + ' |\n')
    print(f'Saved {csv_path} and {md_path}')


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('results', help='experiment_results.json path')
    p.add_argument('--out', default='coevolution_output')
    p.add_argument('--alphas', default='',
                   help='comma-separated α rows for the time-series figure '
                        '(default: endpoints + midpoint)')
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    per_alpha = load(args.results)
    plot_timeseries(per_alpha, pick_alphas(per_alpha, args.alphas), args.out)
    plot_phase(per_alpha, args.out)
    write_summary(per_alpha, args.out)


if __name__ == '__main__':
    main()
