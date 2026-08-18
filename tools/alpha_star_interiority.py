"""Per-seed paired tests of alpha-star interiority and location.

The range-normalised composites locate alpha* on sweep means; this script asks
two sharper questions using the per-seed late-run values (replicate i shares
seed i across alpha levels within a configuration, so contrasts are paired):

1. INTERIORITY: is the bubble composite lower at interior alignment levels
   than at both endpoints (alpha = 0 and alpha = 1), robustly across seeds?
2. LOCATION: do interior candidate levels differ from each other, or is the
   objective flat across the mid-range (which would make the argmin unstable
   across runs/composites, as observed)?

Usage: python3 tools/alpha_star_interiority.py results/switched-sweep/experiment_results.json
"""
import json
import sys

import numpy as np

ALPHAS = [round(0.1 * i, 1) for i in range(11)]


def per_seed_metrics(path):
    with open(path) as f:
        data = json.load(f)
    out = {}
    for res, a in zip(data['all_results'], ALPHAS):
        seci = np.nanmean(
            [res['seci_exploit_ss_runs'], res['seci_explor_ss_runs']], axis=0)
        aeci = np.asarray(res['aeci_var_ss_runs'], dtype=float)
        mae = np.nanmean(
            [res['mae_exploit_ss_runs'], res['mae_explor_ss_runs']], axis=0)
        out[a] = {'seci': seci, 'aeci': aeci, 'mae': mae,
                  'bubble': np.abs(seci) + np.abs(aeci)}
    return out


def paired_ci(x, y):
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    d = d[~np.isnan(d)]
    m = d.mean()
    half = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
    return m, m - half, m + half


def main(path):
    m = per_seed_metrics(path)
    print(f'== {path} ==')
    print('\nPer-seed paired differences of the RAW bubble composite '
          '|SECI|+|AECI-Var| (negative = first level better; '
          '95% CI in brackets; * = CI excludes 0):')
    candidates = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for ref in (0.0, 1.0):
        print(f'\n  vs alpha = {ref}:')
        for a in candidates:
            mean, lo, hi = paired_ci(m[a]['bubble'], m[ref]['bubble'])
            star = '*' if hi < 0 or lo > 0 else ' '
            print(f'    alpha={a}: {mean:+.3f} [{lo:+.3f}, {hi:+.3f}] {star}')
    print('\n  Interior pairwise (flatness of the objective):')
    for a, b in [(0.4, 0.5), (0.5, 0.6), (0.5, 0.8), (0.4, 0.8), (0.6, 0.8)]:
        mean, lo, hi = paired_ci(m[a]['bubble'], m[b]['bubble'])
        star = '*' if hi < 0 or lo > 0 else ' '
        print(f'    {a} vs {b}: {mean:+.3f} [{lo:+.3f}, {hi:+.3f}] {star}')

    print('\n  Component check: paired difference decomposition at the '
          'endpoints (drivers of interiority):')
    for a in (0.5,):
        for ref in (0.0, 1.0):
            ds, lo_s, hi_s = paired_ci(np.abs(m[a]['seci']), np.abs(m[ref]['seci']))
            da, lo_a, hi_a = paired_ci(np.abs(m[a]['aeci']), np.abs(m[ref]['aeci']))
            print(f'    alpha=0.5 vs {ref}:  d|SECI| = {ds:+.3f} '
                  f'[{lo_s:+.3f}, {hi_s:+.3f}],  d|AECI-Var| = {da:+.3f} '
                  f'[{lo_a:+.3f}, {hi_a:+.3f}]')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else
         'results/switched-sweep/experiment_results.json')
