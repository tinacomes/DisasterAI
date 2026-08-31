#!/usr/bin/env python3
"""E1 step 4: population sanity check at the fitted dyadic parameters.

Reruns the population model (test_filter_bubbles machinery, identical
seeding and steady-state conventions) at the E1-fitted acceptance/trust
parameters and checks that the paper's headline structure is intact:

  S1  starvation gradient: explorer MAE rises with alpha (main config)
  S2  operational U: unmet needs at an interior alpha below both
      endpoints (main config)
  S3  structural precondition: exploiter SECI deepens (more negative)
      with alpha under network-bounded access, not under the control

This is a REDUCED grid (default: alpha in {0, 0.6, 1.0}, N=5 seeds) --
a qualitative-structure check, not a replacement for the canonical
N=20 sweep. The full-grid command is printed at the end; run it on CI
for the SI-grade comparison.

Usage:
  python3 experiments/docking_fit/population_check.py \
      [--fitted results-docking-fit/fitted_params.json] \
      [--alphas 0 0.6 1.0] [--n-runs 5] [--configs main control]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

import test_filter_bubbles as tfb  # noqa: E402

CONFIGS = {
    'main': {'mobility': 1, 'network_type': 'spatial_bridged',
             'query_scope': 'network'},
    'control': {'mobility': 0, 'network_type': 'components',
                'query_scope': 'global'},
}
METRICS = ('seci_exploit_mean', 'seci_explor_mean', 'mae_exploit_mean',
           'mae_explor_mean', 'unmet_needs_mean')
# Dyad-fit parameters that carry over to the population model; 'rounds'
# is dyad-only (interaction length) and is not a population parameter.
CARRY = ('d_exploit', 'delta_exploit', 'd_explor', 'delta_explor',
         'initial_trust', 'initial_ai_trust')


def run_cell(params, n_runs, label):
    res = tfb.run_replicated(params, n_runs, label=label)
    return {m: float(tfb.ss(res[m])) for m in METRICS if m in res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fitted', default=os.path.join(
        REPO, 'results-docking-fit', 'fitted_params.json'))
    ap.add_argument('--alphas', nargs='+', type=float,
                    default=[0.0, 0.6, 1.0])
    ap.add_argument('--n-runs', type=int, default=5)
    ap.add_argument('--configs', nargs='+', default=['main', 'control'],
                    choices=['main', 'control'])
    ap.add_argument('--param-set', choices=['fitted', 'default'],
                    default='fitted',
                    help="Which parameter set to run ('default' rebuilds "
                         'the same reduced grid at baseline parameters '
                         'for a like-for-like comparison).')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    with open(args.fitted) as f:
        fitted = json.load(f)
    pset = fitted['fitted_params'] if args.param_set == 'fitted' \
        else fitted['default_params']
    overrides = {k: pset[k] for k in CARRY}
    print(f'Parameter set: {args.param_set} -> {overrides}')

    results = {}
    for config in args.configs:
        for alpha in args.alphas:
            params = {**tfb.base_params, **CONFIGS[config],
                      'ai_alignment_level': alpha, **overrides}
            key = f'{config}|{alpha}'
            results[key] = run_cell(params, args.n_runs, key)
            print(key, {k: round(v, 3) for k, v in results[key].items()})

    checks = {}
    a_lo, a_hi = min(args.alphas), max(args.alphas)
    mid = sorted(args.alphas)[len(args.alphas) // 2]
    if 'main' in args.configs:
        g = lambda a, m: results[f'main|{a}'][m]  # noqa: E731
        checks['S1_starvation_gradient'] = \
            g(a_hi, 'mae_explor_mean') > g(a_lo, 'mae_explor_mean')
        checks['S2_interior_operational_optimum'] = (
            g(mid, 'unmet_needs_mean') < g(a_lo, 'unmet_needs_mean')
            and g(mid, 'unmet_needs_mean') < g(a_hi, 'unmet_needs_mean'))
        checks['S3a_bounded_chamber_persists'] = \
            g(a_hi, 'seci_exploit_mean') < -0.05
    if 'control' in args.configs:
        c = lambda a, m: results[f'control|{a}'][m]  # noqa: E731
        checks['S3b_global_access_dissolves'] = (
            c(a_hi, 'seci_exploit_mean')
            > results[f'main|{a_hi}']['seci_exploit_mean'] + 0.05
            if 'main' in args.configs else None)

    out = {'param_set': args.param_set, 'overrides': overrides,
           'alphas': args.alphas, 'n_runs': args.n_runs,
           'note': 'Reduced-grid qualitative structure check; canonical '
                   'comparison requires the full N=20 sweep.',
           'cells': results, 'checks': checks,
           'all_pass': all(v for v in checks.values() if v is not None)}
    out_path = args.out or os.path.join(
        REPO, 'results-docking-fit',
        f'population_check_{args.param_set}.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    print('\nChecks:', json.dumps(checks, indent=2))
    print(f'ALL PASS: {out["all_pass"]}')
    print(f'Wrote {out_path}')
    print('\nFull SI-grade sweep (CI): run the canonical mechfix commands '
          'with these parameter overrides added to base_params, N=20, '
          'both configurations, all 11 alpha levels.')


if __name__ == '__main__':
    main()
