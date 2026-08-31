#!/usr/bin/env python3
"""E1: effect-size-targeted fit of the dyadic docking micro-parameters.

Fits the model's free dyadic parameters -- acceptance windows (D, delta,
both types) and initial trust levels (human, AI) plus the interaction
length (rounds) -- to dimensionless effect sizes computed from Glickman
& Sharot's published data (targets.json; provenance inside).

Correspondence (stated once, used everywhere):

  * G&S kappa = fraction of the partner-self judgment gap the human has
    adopted by the final collaboration block.
  * Model kappa, AI partner: the truthful AI (alpha = 0) reports the
    ground truth (0) at the implanted false-epicenter cells, so the
    partner-self gap equals the implanted bias fb0 and
        kappa_ai = (fb0 - fbT) / fb0.
    G&S's biased-AI partner and the model's truthful AI both sit at a
    fixed judgment the human does not share; kappa measures adoption of
    the partner's judgment in both cases -- direction is irrelevant to
    the transmission fraction.
  * Model kappa, human partner: the same-type mate holds unimplanted
    beliefs at the false cells (level ~ mate_fb0), so
        kappa_hh = (fb0 - fbT) / (fb0 - mate_fb0).
  * The interaction length (rounds) is a free parameter because G&S
    blocks and model rounds have no common clock.

Orderings enforced as penalties (from the same data):
  * C1/C3: the confirming AI (alpha = 1) retains more of the implanted
    bias than the human partner does.
  * Monotonicity: final bias non-decreasing in alpha (refinement phase,
    Spearman over the alpha grid) -- G&S Exp2's biased vs accurate
    algorithm contrast.

Trust learning rates are NOT fitted: the dyadic harness holds trust
fixed by design (no relief loop, hence no reward signal), so they are
inert here; the trust-side free parameters are the initial trust levels.

Search: Latin-hypercube coarse pass (inverse-CI-weighted quadratic loss)
then a refinement of the top sets on the full alpha grid with more
seeds. Outputs: results-docking-fit/{coarse_grid.csv, refined.csv,
fitted_params.json, fit_report.md, docking_fit.png}.

Usage:
  python3 experiments/docking_fit/fit_docking.py \
      [--n-lhs 96] [--coarse-seeds 8] [--refine-seeds 20] [--top 8]
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import qmc, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from dyadic_docking import run_dyad  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT_DIR = os.path.join(REPO, 'results-docking-fit')

# Parameter box. D/delta bounds bracket the S9 cognitive-profile sweep
# ranges (gap scalar g in [0, 1.5] around D_mid = 3.0, delta_mid = 2.35)
# while preserving the model invariant D_exploit < D_explor,
# delta_exploit > delta_explor by construction of the boxes.
BOX = {
    'd_exploit': (1.0, 3.0),
    'delta_exploit': (2.0, 5.0),
    'd_explor': (3.0, 5.5),
    'delta_explor': (0.6, 2.35),
    'initial_trust': (0.15, 0.6),
    'initial_ai_trust': (0.15, 0.6),
    'rounds': (20, 60),          # snapped to a multiple of 10
}
PARAM_NAMES = list(BOX.keys())

DEFAULTS = {'d_exploit': 2.0, 'delta_exploit': 3.5, 'd_explor': 4.0,
            'delta_explor': 1.2, 'initial_trust': 0.3,
            'initial_ai_trust': 0.25, 'rounds': 40}

TYPES = ('exploitative', 'exploratory')
COARSE_CONDS = (('ai', 0.0), ('ai', 1.0), ('human', None))
REFINE_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
MIN_HH_GAP = 0.3   # guard for the kappa_hh denominator (fb0 - mate_fb0)


def load_targets():
    with open(os.path.join(HERE, 'targets.json')) as f:
        t = json.load(f)['fit_targets']
    se1 = (t['T1_ci95'][1] - t['T1_ci95'][0]) / 3.92
    se2 = (t['T2_ci95'][1] - t['T2_ci95'][0]) / 3.92
    return {'T1': t['T1_kappa_ai'], 'se1': se1,
            'T2': t['T2_kappa_human'], 'se2': se2}


def one_task(task):
    """(point_id, params, agent_type, partner, alpha, seed, rounds) -> row"""
    point_id, params, agent_type, partner, alpha, seed, rounds = task
    overrides = {k: v for k, v in params.items() if k != 'rounds'}
    r = run_dyad(seed, agent_type, partner,
                 alpha if alpha is not None else 0.0, rounds,
                 overrides=overrides)
    return {'point_id': point_id, 'agent_type': agent_type,
            'partner': partner,
            'alpha': alpha if alpha is not None else np.nan,
            'seed': seed, 'fb0': r['false_belief'][0],
            'fbT': r['false_belief'][-1], 'partner_fb0': r['partner_fb0']}


def run_batch(points, conds, n_seeds, pool):
    tasks = []
    for pid, params in points.items():
        rounds = int(params['rounds'])
        for agent_type in TYPES:
            for partner, alpha in conds:
                for seed in range(n_seeds):
                    tasks.append((pid, params, agent_type, partner, alpha,
                                  seed, rounds))
    rows = pool.map(one_task, tasks, chunksize=8)
    return pd.DataFrame(rows)


def point_stats(df):
    """Aggregate one point's runs into the fit statistics."""
    out = {}
    ai0 = df[(df.partner == 'ai') & (df.alpha == 0.0)]
    out['kappa_ai'] = float(((ai0.fb0 - ai0.fbT) / ai0.fb0).mean())
    hh = df[df.partner == 'human'].copy()
    gap = hh.fb0 - hh.partner_fb0
    ok = gap > MIN_HH_GAP
    out['kappa_hh'] = float(((hh.fb0 - hh.fbT)[ok] / gap[ok]).mean()) \
        if ok.any() else np.nan
    ai1 = df[(df.partner == 'ai') & (df.alpha == 1.0)]
    out['retention_a1'] = float((ai1.fbT / ai1.fb0).mean())
    out['retention_hh'] = float((hh.fbT / hh.fb0).mean())
    # per-alpha mean final bias per type (refinement monotonicity)
    ai = df[df.partner == 'ai']
    rho = []
    for t in TYPES:
        m = ai[ai.agent_type == t].groupby('alpha').fbT.mean()
        if len(m) >= 3:
            rho.append(spearmanr(m.index.values, m.values).statistic)
    out['mono_rho'] = float(np.mean(rho)) if rho else np.nan
    return out


def loss(stats, tg):
    L = ((stats['kappa_ai'] - tg['T1']) / tg['se1']) ** 2
    if not np.isnan(stats['kappa_hh']):
        L += ((stats['kappa_hh'] - tg['T2']) / tg['se2']) ** 2
    else:
        L += 25.0
    L += 25.0 * max(0.0, stats['retention_hh'] - stats['retention_a1']) ** 2
    if not np.isnan(stats.get('mono_rho', np.nan)):
        L += 25.0 * max(0.0, 0.99 - stats['mono_rho']) ** 2
    return float(L)


def lhs_points(n, rng_seed=7):
    sampler = qmc.LatinHypercube(d=len(PARAM_NAMES), seed=rng_seed)
    unit = sampler.random(n)
    lo = np.array([BOX[k][0] for k in PARAM_NAMES])
    hi = np.array([BOX[k][1] for k in PARAM_NAMES])
    pts = {}
    for i, row in enumerate(qmc.scale(unit, lo, hi)):
        p = dict(zip(PARAM_NAMES, [float(v) for v in row]))
        p['rounds'] = int(round(p['rounds'] / 10) * 10)
        pts[f'lhs{i:03d}'] = p
    pts['default'] = dict(DEFAULTS)
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-lhs', type=int, default=96)
    ap.add_argument('--coarse-seeds', type=int, default=8)
    ap.add_argument('--refine-seeds', type=int, default=20)
    ap.add_argument('--top', type=int, default=8)
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    tg = load_targets()
    print(f'Targets: kappa_ai {tg["T1"]} (se {tg["se1"]:.3f}), '
          f'kappa_hh {tg["T2"]} (se {tg["se2"]:.3f})')

    points = lhs_points(args.n_lhs)
    with Pool(args.workers) as pool:
        print(f'Coarse pass: {len(points)} points x 2 types x '
              f'{len(COARSE_CONDS)} conditions x {args.coarse_seeds} seeds')
        coarse = run_batch(points, COARSE_CONDS, args.coarse_seeds, pool)
        coarse.to_csv(os.path.join(OUT_DIR, 'coarse_grid.csv'), index=False)

        rows = []
        for pid, params in points.items():
            st = point_stats(coarse[coarse.point_id == pid])
            rows.append({'point_id': pid, **params, **st,
                         'loss': loss(st, tg)})
        cdf = pd.DataFrame(rows).sort_values('loss')
        print(cdf.head(10).to_string(index=False))

        keep = list(cdf.head(args.top).point_id)
        if 'default' not in keep:
            keep.append('default')
        refine_points = {pid: points[pid] for pid in keep}
        refine_conds = tuple(('ai', a) for a in REFINE_ALPHAS) \
            + (('human', None),)
        print(f'\nRefinement: {len(refine_points)} points x 2 types x '
              f'{len(refine_conds)} conditions x {args.refine_seeds} seeds')
        refined = run_batch(refine_points, refine_conds,
                            args.refine_seeds, pool)
        refined.to_csv(os.path.join(OUT_DIR, 'refined.csv'), index=False)

    rrows = []
    for pid in refine_points:
        st = point_stats(refined[refined.point_id == pid])
        rrows.append({'point_id': pid, **refine_points[pid], **st,
                      'loss': loss(st, tg)})
    rdf = pd.DataFrame(rrows).sort_values('loss')
    print(rdf.to_string(index=False))

    best_id = rdf[rdf.point_id != 'default'].iloc[0].point_id
    best = refine_points[best_id]
    best_stats = point_stats(refined[refined.point_id == best_id])
    default_stats = point_stats(refined[refined.point_id == 'default'])

    fitted = {
        'targets': tg,
        'fitted_params': best, 'fitted_point_id': best_id,
        'fitted_stats': best_stats,
        'fitted_loss': loss(best_stats, tg),
        'default_params': DEFAULTS, 'default_stats': default_stats,
        'default_loss': loss(default_stats, tg),
        'search': {'n_lhs': args.n_lhs, 'coarse_seeds': args.coarse_seeds,
                   'refine_seeds': args.refine_seeds, 'box': BOX},
        'top10_coarse': cdf.head(10).to_dict('records'),
    }
    with open(os.path.join(OUT_DIR, 'fitted_params.json'), 'w') as f:
        json.dump(fitted, f, indent=2)

    write_report(fitted, cdf, rdf)
    make_figure(refined, best_id, tg)
    print(f'\nWrote {OUT_DIR}/fitted_params.json, fit_report.md, '
          f'docking_fit.png')


def write_report(fitted, cdf, rdf):
    tg = fitted['targets']
    b, d = fitted['fitted_stats'], fitted['default_stats']
    top = cdf.head(10)
    lines = [
        '# E1 docking fit report', '',
        'Targets (Glickman & Sharot 2025, computed from published data; '
        'see `experiments/docking_fit/targets.json` for provenance):', '',
        f'- kappa_ai (transmission from AI partner): **{tg["T1"]}** '
        f'(se {tg["se1"]:.3f})',
        f'- kappa_human (transmission from human partner): **{tg["T2"]}** '
        f'(se {tg["se2"]:.3f})', '',
        '| quantity | target | default params | fitted params |',
        '|---|---|---|---|',
        f'| kappa_ai | {tg["T1"]} | {d["kappa_ai"]:.3f} | '
        f'{b["kappa_ai"]:.3f} |',
        f'| kappa_human | {tg["T2"]} | {d["kappa_hh"]:.3f} | '
        f'{b["kappa_hh"]:.3f} |',
        f'| retention alpha=1 (> retention human) | ordering | '
        f'{d["retention_a1"]:.3f} vs {d["retention_hh"]:.3f} | '
        f'{b["retention_a1"]:.3f} vs {b["retention_hh"]:.3f} |',
        f'| monotonicity rho(final bias, alpha) | ~1 | '
        f'{d["mono_rho"]:.3f} | {b["mono_rho"]:.3f} |',
        f'| loss | -- | {fitted["default_loss"]:.2f} | '
        f'{fitted["fitted_loss"]:.2f} |', '',
        'Fitted parameters (defaults in brackets):', '',
    ]
    for k, v in fitted['fitted_params'].items():
        dv = fitted['default_params'][k]
        vv = f'{v:.2f}' if isinstance(v, float) else str(v)
        lines.append(f'- `{k}` = **{vv}** (default {dv})')
    lines += [
        '', '## Identifiability',
        '', 'Coarse-pass top 10 (loss-ranked); parameter ranges within '
        'this set indicate ridge directions -- the fit constrains '
        'combinations, not every coordinate:', '',
        top[PARAM_NAMES + ['loss']].round(3).to_markdown(index=False), '',
        'Trust learning rates are not identified by the dyad (trust is '
        'held fixed there by design); the fitted trust-side quantities '
        'are the initial trust levels.', '',
        '## Refined comparison (all refined points)', '',
        rdf[['point_id'] + PARAM_NAMES + ['kappa_ai', 'kappa_hh',
            'retention_a1', 'retention_hh', 'mono_rho', 'loss']]
        .round(3).to_markdown(index=False), '',
    ]
    with open(os.path.join(OUT_DIR, 'fit_report.md'), 'w') as f:
        f.write('\n'.join(lines))


def make_figure(refined, best_id, tg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # (a) kappa comparison
    ax = axes[0]
    labels = ['κ AI partner', 'κ human partner']
    targets = [tg['T1'], tg['T2']]
    errs = [1.96 * tg['se1'], 1.96 * tg['se2']]
    for i, (pid, color, name) in enumerate(
            (('default', '#888888', 'model, default'),
             (best_id, '#B45F06', 'model, fitted'))):
        df = refined[refined.point_id == pid]
        st = point_stats(df)
        ax.bar(np.arange(2) + (i - 0.5) * 0.28,
               [st['kappa_ai'], st['kappa_hh']], width=0.26,
               color=color, label=name)
    ax.errorbar(np.arange(2) + 0.0, targets, yerr=errs, fmt='k_',
                markersize=18, capsize=5, lw=1.8,
                label='Glickman & Sharot (95% CI)', zorder=5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_ylabel('Transmission of partner judgment')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')
    ax.set_title('(a) Effect-size targets vs model')

    # (b) final bias vs alpha, default vs fitted
    ax = axes[1]
    for pid, color, name in (('default', '#888888', 'default'),
                             (best_id, '#B45F06', 'fitted')):
        df = refined[(refined.point_id == pid) & (refined.partner == 'ai')]
        for t, ls in (('exploitative', '-'), ('exploratory', '--')):
            m = df[df.agent_type == t].groupby('alpha').fbT.agg(
                ['mean', 'sem'])
            ax.errorbar(m.index, m['mean'], yerr=m['sem'], fmt=f'{ls}o',
                        color=color, capsize=3, ms=4,
                        label=f'{name}, {t[:7]}')
        hh = refined[(refined.point_id == pid)
                     & (refined.partner == 'human')]
        ax.axhline(hh.fbT.mean(), color=color, ls=':', lw=1.2)
    ax.set_xlabel('AI alignment α')
    ax.set_ylabel('Final false-belief strength')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title('(b) Dyadic dose-response, default vs fitted\n'
                 '(dotted: human-human endpoint)')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'docking_fit.png'), dpi=150,
                bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
