#!/usr/bin/env python3
"""M3 boundary strengthening: paired per-seed deltas at the high-alpha
boundary (alpha in {0.8, 0.9, 1.0}) with N=50 replications.

Consumes the per-alpha worker JSONs produced by run-boundary-sweep.yml
(test_filter_bubbles.py --single-alpha), one per (configuration x alpha):

    <root>/**/boundary-baseline-*/bubble_alpha_<a>.json   (control)
    <root>/**/boundary-switches-*/bubble_alpha_<a>.json   (main model)

Any directory layout works as long as the path of each JSON contains either
'baseline' or 'switches'/'switched'.

For every outcome (SECI per type, AECI-IE per type, MAE per type, unmet
needs, precision per type) it reports the paired per-seed delta
(main - control) at each boundary alpha with a t-based 95% CI and
Holm-corrected p-values (within outcome, across the boundary levels) —
the firm CIs behind the Finding-1 paired deltas and the alpha >= 0.9 claims.

Usage:
    python3 tools/boundary_deltas.py <root> [save_dir]

Outputs: boundary_deltas.md, boundary_deltas.csv (into save_dir, default '.').
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_filter_bubbles import _normalize_result_conventions  # noqa: E402
from tools.sweep_regression import holm  # noqa: E402

OUTCOMES = [
    ('SECI (exploit)',      'seci_exploit_ss_runs'),
    ('SECI (explor)',       'seci_explor_ss_runs'),
    ('AECI-IE (exploit)',   'aeci_ie_exploit_ss_runs'),
    ('AECI-IE (explor)',    'aeci_ie_explor_ss_runs'),
    ('MAE (exploit)',       'mae_exploit_ss_runs'),
    ('MAE (explor)',        'mae_explor_ss_runs'),
    ('Unmet needs',         'unmet_needs_ss_runs'),
    ('Precision (exploit)', 'prec_exploit_ss_runs'),
    ('Precision (explor)',  'prec_explor_ss_runs'),
]


def config_of(path):
    p = path.lower()
    if 'baseline' in p:
        return 'baseline'
    if 'switches' in p or 'switched' in p:
        return 'switches'
    return None


def load(root):
    data = {}   # (config, alpha) -> result dict
    for path in sorted(glob.glob(os.path.join(root, '**', 'bubble_alpha_*.json'),
                                 recursive=True)):
        cfg = config_of(path)
        if cfg is None:
            print(f'  Skipping (no config in path): {path}')
            continue
        with open(path) as f:
            d = _normalize_result_conventions(json.load(f))
        data[(cfg, float(d['alpha']))] = d['result']
    return data


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else '.'
    save_dir = sys.argv[2] if len(sys.argv) > 2 else '.'
    from scipy import stats

    data = load(root)
    alphas = sorted({a for _, a in data})
    if not alphas:
        print(f'No bubble_alpha_*.json files found under {root!r}')
        sys.exit(1)
    print(f'Boundary levels found: {alphas}')

    os.makedirs(save_dir, exist_ok=True)
    md_path = os.path.join(save_dir, 'boundary_deltas.md')
    csv_path = os.path.join(save_dir, 'boundary_deltas.csv')
    csv_rows = [('outcome', 'alpha', 'delta', 'ci_lo', 'ci_hi', 'p_holm', 'n')]

    with open(md_path, 'w') as f:
        f.write('# Boundary paired deltas (M3): main model − control at '
                'high α\n\n')
        f.write('Per-seed paired deltas of the late-run (last 75 ticks) '
                'outcome means; replicate *i* shares seed *i* across '
                'configurations. 95% CI from the paired t distribution; '
                'p-values Holm-corrected within each outcome across the '
                'boundary levels. Bold = adjusted p < 0.05.\n\n')
        for out_label, key in OUTCOMES:
            rows, pvals = [], []
            for a in alphas:
                r0 = np.asarray(data.get(('baseline', a), {}).get(key, []),
                                dtype=float)
                r1 = np.asarray(data.get(('switches', a), {}).get(key, []),
                                dtype=float)
                if len(r0) == 0 or len(r0) != len(r1):
                    continue
                d = r1 - r0
                d = d[~np.isnan(d)]
                if len(d) < 2:
                    continue
                n = len(d)
                mean = float(d.mean())
                se = float(d.std(ddof=1) / np.sqrt(n))
                tcrit = stats.t.ppf(0.975, n - 1)
                p = (float(stats.ttest_1samp(d, 0.0).pvalue)
                     if se > 1e-15 else (0.0 if abs(mean) > 1e-15 else 1.0))
                rows.append([a, mean, mean - tcrit * se, mean + tcrit * se, p, n])
                pvals.append(p)
            f.write(f'### {out_label}\n\n')
            if not rows:
                f.write('*not pairable / metric absent*\n\n')
                continue
            for row, p_adj in zip(rows, holm(pvals)):
                row[4] = float(p_adj)
            f.write('| α | Δ (main − control) | 95% CI | p (Holm) | n |\n')
            f.write('|---|---|---|---|---|\n')
            for a, mean, lo, hi, p, n in rows:
                mark = '**' if p < 0.05 else ''
                f.write(f'| {a:.1f} | {mark}{mean:+.3f}{mark} | '
                        f'[{lo:+.3f}, {hi:+.3f}] | {p:.3g} | {n} |\n')
                csv_rows.append((out_label, a, f'{mean:.6g}', f'{lo:.6g}',
                                 f'{hi:.6g}', f'{p:.3g}', n))
            f.write('\n')

    with open(csv_path, 'w') as f:
        for row in csv_rows:
            f.write(','.join(str(c) for c in row) + '\n')
    print(f'Boundary tables saved: {md_path}, {csv_path}')
    with open(md_path) as f:
        print('\n' + f.read())


if __name__ == '__main__':
    main()
