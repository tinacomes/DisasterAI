#!/usr/bin/env python3
"""M9 alpha-reversal (hysteresis) experiment: do echo chambers outlive the
alignment that created them?

Motivated by the lifecycle finding that accuracy-seeking chambers formed at
alpha >= 0.6 never dissolve within the 200-tick horizon: is that only
censoring (the run ends too early) or genuine irreversibility? Here the AI
policy itself is repaired mid-run — alpha switches from ``alpha_pre`` to
``alpha_post`` at ``switch_tick`` (the AI reads the model attribute live,
so the change is immediate) — and the question is whether the chambers,
belief starvation, and periphery gaps built under ``alpha_pre`` unwind
once the sycophancy is removed.

Anchors: the same script with ``alpha_pre == alpha_post`` (the switch is a
no-op) gives the constant-policy reference trajectories on the SAME seeds,
so every comparison is seed-paired.

Readout (``collect`` mode), per seed and per condition:
  * whether the accuracy-seekers' chamber (SECI < -0.1) is standing at the
    switch tick, and if so whether — and with what lag — it dissolves
    afterwards (sustained > -0.05, the thresholds of test_filter_bubbles);
  * endpoint (last-75-tick) SECI, MAE, precision, L1+ pool, and spatial
    periphery gaps, compared seed-paired against BOTH anchors: recovery to
    the truthful-anchor level = reversible; endpoint stuck near the
    aligned-anchor level = hysteresis.

Hysteresis verdict: chambers (or gaps) that persist after the policy
repair demonstrate that the damage is a property of the human network
state, not of the ongoing AI policy — the strongest dynamic claim the
paper could make. Symmetric probe: ``--alpha-pre 0.0 --alpha-post 1.0``
tests late-onset formation.

Usage (one condition per CI matrix job; main-model configuration default):
  python3 experiments/alpha_reversal.py run \
      --alpha-pre 1.0 --alpha-post 0.0 --switch-tick 100 --ticks 300 \
      --n-runs 20 --outfile out/reversal_1.0_to_0.0.json
  python3 experiments/alpha_reversal.py collect --results-dir out \
      --save-dir reversal_plots
"""

import argparse
import glob
import json
import os
import random
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'tools'))

from lifecycle_metrics import (chamber_episodes, _arr,        # noqa: E402
                               FORM_THRESH)

# Curated per-run series kept in the output JSON (per-seed trajectories are
# the point of this experiment — they enable CIs on the recovery claims).
RUN_KEYS = [
    'metric_ticks',
    'seci_exploit', 'seci_explor', 'seci_pop',
    'mae_exploit', 'mae_explor', 'prec_exploit', 'prec_explor',
    'l1pool_explor', 'l1pool_exploit', 'lockin_explor',
    'ai_query_ratio_exploit', 'ai_query_ratio_explor',
    'trust_ai_exploit', 'trust_ai_explor',
    'periph_sp_mae_gap', 'periph_sp_aid_gap',
    'unmet_needs',
]

SS_SAMPLES = 15   # steady-state window: last 15 samples = 75 ticks


# ------------------------------ run mode -----------------------------------

def cmd_run(args):
    import test_filter_bubbles as tfb   # heavy import only in run mode

    params = {**tfb.base_params,
              'ai_alignment_level': args.alpha_pre,
              'ticks': args.ticks,
              'mobility': args.mobility,
              'network_type': args.network_type,
              'query_scope': args.query_scope}
    if args.salience_weight is not None:
        params['salience_weight'] = args.salience_weight

    schedule = None
    if args.alpha_post != args.alpha_pre:
        schedule = {args.switch_tick: args.alpha_post}

    label = (f'alpha {args.alpha_pre} -> {args.alpha_post} '
             f'@tick {args.switch_tick}' if schedule
             else f'constant alpha {args.alpha_pre} (anchor)')
    print(f'Reversal condition: {label}; ticks={args.ticks}, '
          f'n_runs={args.n_runs}, seeds {args.seed_base}..'
          f'{args.seed_base + args.n_runs - 1}')

    runs = []
    for i in range(args.n_runs):
        print(f'  Replicate {i + 1}/{args.n_runs}...')
        random.seed(args.seed_base + i)
        np.random.seed(args.seed_base + i)
        full = tfb.run_one_sim(params, alpha_schedule=schedule)
        runs.append({k: full[k] for k in RUN_KEYS if k in full})

    out = {
        'condition': {
            'alpha_pre': args.alpha_pre,
            'alpha_post': args.alpha_post,
            'switch_tick': args.switch_tick,
            'ticks': args.ticks,
            'seed_base': args.seed_base,
            'mobility': args.mobility,
            'network_type': args.network_type,
            'query_scope': args.query_scope,
            'salience_weight': args.salience_weight,
        },
        'n_runs': args.n_runs,
        'conventions': {'ie_sign': 'negative_echo'},
        'runs': runs,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.outfile)), exist_ok=True)
    with open(args.outfile, 'w') as f:
        json.dump(out, f)
    print(f'Saved {len(runs)} runs -> {args.outfile}')


# ---------------------------- collect mode ---------------------------------

def _cond_label(c):
    if c['alpha_pre'] == c['alpha_post']:
        return f"constant α={c['alpha_pre']:.1f}"
    return f"α={c['alpha_pre']:.1f}→{c['alpha_post']:.1f} @t={c['switch_tick']}"


def _ss(series, n=SS_SAMPLES):
    a = _arr(series)
    return float(np.nanmean(a[-min(n, len(a)):]))


def _mean_ci(vals, conf=1.96):
    a = np.asarray([v for v in vals if v is not None and not np.isnan(v)])
    if len(a) == 0:
        return float('nan'), float('nan')
    se = a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else float('nan')
    return float(a.mean()), float(conf * se)


def post_switch_recovery(run, switch_tick, horizon):
    """Per-seed: was the explorer chamber standing at the switch, and did the
    episode containing the switch dissolve afterwards (lag in ticks)?"""
    ticks = run['metric_ticks']
    eps = chamber_episodes(run['seci_explor'], ticks, horizon)
    current = next((e for e in eps if e[0] <= switch_tick < e[1]), None)
    if current is None:
        return {'standing_at_switch': False, 'dissolved_after': None,
                'lag': None}
    start, end, censored = current
    return {'standing_at_switch': True,
            'dissolved_after': not censored,
            'lag': (end - switch_tick) if not censored else None}


def cmd_collect(args):
    files = sorted(glob.glob(os.path.join(args.results_dir, '**',
                                          'reversal_*.json'),
                             recursive=True))
    if not files:
        raise SystemExit(f'No reversal_*.json under {args.results_dir}')
    conds = []
    for path in files:
        with open(path) as f:
            conds.append(json.load(f))
    conds.sort(key=lambda c: (c['condition']['alpha_pre'],
                              c['condition']['alpha_post']))
    os.makedirs(args.save_dir, exist_ok=True)

    # ---- per-seed recovery + endpoint stats per condition ----
    # Anchors are evaluated at the switch tick of the switch conditions, so
    # "dissolved after" is directly comparable: it answers whether a chamber
    # standing at that tick dissolves spontaneously under the constant policy.
    switch_ticks = {c['condition']['switch_tick'] for c in conds
                    if c['condition']['alpha_pre'] != c['condition']['alpha_post']}
    default_switch = min(switch_ticks) if switch_ticks else None
    rows = []
    for c in conds:
        cond = c['condition']
        horizon = cond['ticks']
        is_anchor = cond['alpha_pre'] == cond['alpha_post']
        switch = default_switch if is_anchor else cond['switch_tick']
        rec = [post_switch_recovery(r, switch, horizon) for r in c['runs']] \
            if switch is not None else []
        standing = [r for r in rec if r['standing_at_switch']]
        dissolved = [r for r in standing if r['dissolved_after']]
        lags = [r['lag'] for r in dissolved]
        row = {'label': _cond_label(cond), 'n': c['n_runs'],
               'seed_base': cond.get('seed_base', 0)}
        if switch is not None:
            row.update({
                'standing_at_switch': f'{len(standing)}/{len(rec)}',
                'dissolved_after': f'{len(dissolved)}/{len(standing)}'
                                    if standing else '--',
                'median_lag': float(np.median(lags)) if lags else None,
            })
        else:
            row.update({'standing_at_switch': '--', 'dissolved_after': '--',
                        'median_lag': None})
        for key, name in (('seci_explor', 'SECI_explor'),
                          ('seci_exploit', 'SECI_exploit'),
                          ('mae_explor', 'MAE_explor'),
                          ('l1pool_explor', 'L1pool_explor'),
                          ('prec_explor', 'prec_explor'),
                          ('periph_sp_mae_gap', 'sp_MAE_gap'),
                          ('periph_sp_aid_gap', 'sp_aid_gap')):
            m, ci = _mean_ci([_ss(r[key]) for r in c['runs']])
            row[f'end_{name}'] = f'{m:+.3f} ± {ci:.3f}'
        rows.append(row)

    md_path = os.path.join(args.save_dir, 'reversal_summary.md')
    with open(md_path, 'w') as f:
        f.write('# Alpha-reversal (hysteresis) experiment — summary\n\n')
        f.write('Per-seed post-switch recovery of the accuracy-seekers\' '
                'chamber (standing = SECI episode containing the switch '
                'tick; dissolution thresholds as in test_filter_bubbles), '
                'plus endpoint (last-75-tick) means ± 95% CI. Compare each '
                'switch row against BOTH constant anchors on the same '
                'seeds: recovery to the truthful anchor = reversible; '
                'endpoint at the aligned anchor = hysteresis.\n\n')
        cols = ['label', 'n', 'standing_at_switch', 'dissolved_after',
                'median_lag', 'end_SECI_explor', 'end_SECI_exploit',
                'end_MAE_explor', 'end_L1pool_explor', 'end_prec_explor',
                'end_sp_MAE_gap', 'end_sp_aid_gap']
        f.write('| ' + ' | '.join(cols) + ' |\n')
        f.write('|---' * len(cols) + '|\n')
        for row in rows:
            f.write('| ' + ' | '.join(
                '--' if row.get(k) is None else str(row.get(k, '--'))
                for k in cols) + ' |\n')
    print(f'wrote {md_path}')

    # ---- trajectory figure ----
    panels = [('seci_explor', 'SECI accuracy-seeking', 'metric'),
              ('seci_exploit', 'SECI confirmation-seeking', 'metric'),
              ('mae_explor', 'MAE accuracy-seeking', 'metric'),
              ('l1pool_explor', 'L1+ pool accuracy-seeking', 'metric'),
              ('periph_sp_aid_gap', 'spatial aid gap (far − near)', 'metric'),
              ('ai_query_ratio_exploit', 'AI query share confirm.-seeking',
               'tick')]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    cmap = plt.cm.tab10
    for ci_, c in enumerate(conds):
        cond = c['condition']
        is_anchor = cond['alpha_pre'] == cond['alpha_post']
        color = cmap(ci_ % 10)
        for ax, (key, title, axis) in zip(axes.flat, panels):
            series = [_arr(r[key]) for r in c['runs']]
            n = min(len(s) for s in series)
            arr = np.vstack([s[:n] for s in series])
            x = (c['runs'][0]['metric_ticks'][:n] if axis == 'metric'
                 else list(range(n)))
            mean = np.nanmean(arr, axis=0)
            se = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])
            ls = '--' if is_anchor else '-'
            ax.plot(x, mean, ls, color=color, lw=1.8,
                    label=_cond_label(cond))
            ax.fill_between(x, mean - 1.96 * se, mean + 1.96 * se,
                            color=color, alpha=0.12)
            if not is_anchor:
                ax.axvline(cond['switch_tick'], color=color, ls=':', lw=1)
    for ax, (key, title, _) in zip(axes.flat, panels):
        ax.set_title(title)
        ax.set_xlabel('Simulation tick')
        ax.grid(alpha=0.3)
        if key.startswith('seci'):
            ax.axhline(FORM_THRESH, color='grey', ls=':', lw=1)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle('Alpha-reversal experiment: dotted vertical line = policy '
                 'switch; dashed = constant-α anchors (same seeds); '
                 'shading = 95% CI')
    fig.tight_layout()
    fig_path = os.path.join(args.save_dir, 'reversal_trajectories.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {fig_path}')


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = p.add_subparsers(dest='cmd', required=True)

    r = sub.add_parser('run', help='run one reversal condition')
    r.add_argument('--outfile', required=True)
    r.add_argument('--alpha-pre', type=float, required=True,
                   help='alignment level before the switch')
    r.add_argument('--alpha-post', type=float, required=True,
                   help='alignment level from switch-tick on (== alpha-pre '
                        'for a constant anchor)')
    r.add_argument('--switch-tick', type=int, default=100)
    r.add_argument('--ticks', type=int, default=300)
    r.add_argument('--n-runs', type=int, default=20)
    r.add_argument('--seed-base', type=int, default=0)
    r.add_argument('--mobility', type=int, choices=[0, 1], default=1,
                   help='default 1: main-model configuration')
    r.add_argument('--network-type', default='spatial_bridged',
                   choices=['components', 'spatial_bridged',
                            'spatial_smallworld'])
    r.add_argument('--query-scope', default='network',
                   choices=['global', 'network'])
    r.add_argument('--salience-weight', type=float, default=None)
    r.set_defaults(func=cmd_run)

    c = sub.add_parser('collect', help='aggregate reversal_*.json + plot')
    c.add_argument('--results-dir', required=True)
    c.add_argument('--save-dir', required=True)
    c.set_defaults(func=cmd_collect)

    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
