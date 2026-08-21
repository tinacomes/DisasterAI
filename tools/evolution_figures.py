#!/usr/bin/env python3
"""Time-evolution figures for the echo indices, per agent type.

Steady-state endpoints alone are a poor summary: they hide whether an
index is still moving at T, whether a chamber formed and recovered, and
how much of the value is transient. This plots the full 200-tick
trajectory of every index, split by agent type, with one line per
alignment level and the steady-state window shaded, so the endpoint can
be read in the context of the path that produced it.

Usage:
    python3 tools/evolution_figures.py <experiment_results.json> [--out DIR]
        [--label NAME] [--alphas 0.0,0.3,0.5,0.7,0.9,1.0]
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# rows: (key stem, display name, sign note, y-reference line)
ROWS = [
    ('seci',        'SECI\n(social echo)',            'negative = chamber', 0.0),
    ('aeci_ie',     'AECI-IE\n(AI channel vs global)', 'negative = chamber', 0.0),
    ('aeci_ie_rel', 'AECI-IE-rel\n(AI vs own beliefs)', '0 = channel mirrors community', 0.0),
    ('mae',         'Belief error\n(MAE, affected cells)', 'lower is better', None),
    ('l1pool',      'L1+ belief pool\n(per agent)',    'information held',  None),
]
TYPES = [('exploit', 'Exploitative — confirmation-seeking'),
         ('explor',  'Exploratory — accuracy-seeking')]


def series(res, key):
    v = res.get(key + '_mean')
    return np.array(v, dtype=float) if isinstance(v, list) and v else None


def build(path, out_dir, label, alphas_want, ss_frac=0.375):
    d = json.load(open(path))
    A, ar = d['alignment_sweep'], d['all_results']
    ticks = np.array(ar[0].get('metric_ticks', range(len(ar[0].get('seci_exploit_mean', [])))),
                     dtype=float)
    sel = [(a, r) for a, r in zip(A, ar) if any(abs(a - w) < 1e-9 for w in alphas_want)]
    if not sel:
        sel = list(zip(A, ar))
    cmap = plt.get_cmap('viridis')
    norm = lambda a: cmap(0.08 + 0.84 * a)

    fig, axes = plt.subplots(len(ROWS), 2, figsize=(12.4, 2.45 * len(ROWS)),
                             sharex=True)
    ss_start = ticks[-1] * (1 - ss_frac)
    for ri, (stem, name, note, ref) in enumerate(ROWS):
        for ci, (tkey, tname) in enumerate(TYPES):
            ax = axes[ri, ci]
            ax.axvspan(ss_start, ticks[-1], color='0.5', alpha=0.10, lw=0)
            if ref is not None:
                ax.axhline(ref, color='k', ls=':', lw=.8, alpha=.45)
            any_data = False
            for a, r in sel:
                y = series(r, f'{stem}_{tkey}')
                if y is None or len(y) != len(ticks):
                    continue
                ax.plot(ticks, y, color=norm(a), lw=1.5, alpha=.95)
                any_data = True
            if not any_data:
                ax.text(.5, .5, 'not recorded', ha='center', va='center',
                        transform=ax.transAxes, color='0.6', fontsize=9)
            if ri == 0:
                ax.set_title(tname, fontsize=11, pad=9)
            if ci == 0:
                ax.set_ylabel(name, fontsize=9.5)
            ax.tick_params(labelsize=8.5)
            ax.grid(alpha=.16, lw=.6)
            if stem in ('seci', 'aeci_ie', 'aeci_ie_rel'):
                ax.set_ylim(-1.02, .45)
            ax.text(.985, .06 if stem == 'mae' else .94, note, transform=ax.transAxes,
                    ha='right', va='bottom' if stem == 'mae' else 'top',
                    fontsize=7.6, color='0.42')
    for ax in axes[-1]:
        ax.set_xlabel('tick', fontsize=9.5)

    handles = [Line2D([], [], color=norm(a), lw=2, label=f'α = {a:.1f}') for a, _ in sel]
    handles.append(Line2D([], [], color='0.5', lw=8, alpha=.3,
                          label=f'steady-state window (last {int(ss_frac*100)}%)'))
    fig.legend(handles=handles, loc='lower center', ncol=len(handles),
               frameon=False, fontsize=9, bbox_to_anchor=(.5, -.012))
    fig.suptitle(f'Evolution of the echo indices by agent type — {label}',
                 fontsize=12.5, y=.997)
    fig.tight_layout(rect=[0, .035, 1, .985])
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f'evolution_{label.replace(" ", "_")}.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    print('saved', p)
    return p


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('results')
    ap.add_argument('--out', default='.')
    ap.add_argument('--label', default='sweep')
    ap.add_argument('--alphas', default='0.0,0.3,0.5,0.7,0.9,1.0')
    a = ap.parse_args()
    build(a.results, a.out, a.label,
          [float(x) for x in a.alphas.split(',') if x.strip()])
