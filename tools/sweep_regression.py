#!/usr/bin/env python3
"""M4 statistics upgrade: mixed-effects regression + Holm-corrected paired
contrasts over the paired two-configuration alignment sweep.

Per outcome (SECI per type, AECI-IE per type, MAE per type, unmet needs,
precision per type) this script fits

    outcome_z ~ alpha + alpha^2 + config + alpha:config + alpha^2:config

with the replicate seed as grouping factor (statsmodels MixedLM with a random
intercept per seed; falls back to seed-clustered OLS when the mixed model
fails to converge). Outcomes are z-scored across the full pooled sample, so
coefficients are standardized effects (SD units per unit alpha / per
configuration switch). alpha enters raw on [0, 1]; config is 0 = control
(global access) and 1 = main model (network-bounded).

It also replaces the raw per-level delta CIs with Holm-corrected paired
contrasts: at each alpha level the per-seed delta (main - control) is tested
with a paired t-test, and the p-values are Holm-corrected within each outcome
family (11 levels). Replicate i is seeded with seed i in both configurations,
so the pairing is exact.

Input layout (the download-artifact tree of compare-network-mobility.yml,
identical to tools/compare_configs.py):

    <root>/plots-config-baseline/experiment_results.json   (control)
    <root>/plots-config-switches/experiment_results.json   (main model)

Usage:
    python3 tools/sweep_regression.py <root> [save_dir]

Outputs (into save_dir, default '.'):
    regression_table.md    SI-ready Markdown: standardized mixed-model
                           coefficients per outcome + Holm-corrected paired
                           contrasts per alpha level
    regression_table.csv   the same coefficients, machine-readable
"""
import glob
import os
import sys
import warnings

import numpy as np

# Ensure the repo root (parent of tools/) is importable regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_filter_bubbles import load_results, ALIGNMENT_SWEEP  # noqa: E402

# Outcome label → key of the per-replicate late-run lists in the aggregated
# result dicts (the *_ss_runs values: late-run mean of the last 75 ticks per
# replicate). Per-type reporting is the default (PNAS brief §5); unmet needs
# is a population-level outcome.
OUTCOMES = [
    ('SECI (exploit)',      'seci_exploit_ss_runs'),
    ('SECI (explor)',       'seci_explor_ss_runs'),
    ('AECI-IE (exploit)',   'aeci_ie_exploit_ss_runs'),
    ('AECI-IE (explor)',    'aeci_ie_explor_ss_runs'),
    ('AECI-IE-chan (exploit)', 'aeci_ie_chan_exploit_ss_runs'),
    ('AECI-IE-chan (explor)',  'aeci_ie_chan_explor_ss_runs'),
    # Population-level (societal) series — present since commit 30f89e0;
    # absent keys degrade to skipped outcomes for older archives.
    ('SECI (population)',        'seci_pop_ss_runs'),
    ('AECI-IE-chan (population)', 'aeci_ie_chan_pop_ss_runs'),
    ('MAE (exploit)',       'mae_exploit_ss_runs'),
    ('MAE (explor)',        'mae_explor_ss_runs'),
    ('Unmet needs',         'unmet_needs_ss_runs'),
    ('Precision (exploit)', 'prec_exploit_ss_runs'),
    ('Precision (explor)',  'prec_explor_ss_runs'),
]

# Configuration directory label → (display name, config dummy). The dummy
# encodes the paper's contrast: 0 = control (global access), 1 = main model
# (network-bounded access).
CONFIG_CODES = {
    'baseline': ('control (global access)', 0),
    'switches': ('main model (network-bounded)', 1),
    'switched': ('main model (network-bounded)', 1),
}


def setting_label(path):
    parent = os.path.basename(os.path.dirname(path))
    for prefix in ('plots-config-', 'plots-'):
        if parent.startswith(prefix):
            return parent[len(prefix):]
    return parent


def collect_long(paths):
    """Long-format records: one row per (config, alpha, seed, outcome).

    Returns {outcome_label: dict with arrays alpha, config, seed, value} and
    the two display names ordered (control, main).
    """
    tables = {label: {'alpha': [], 'config': [], 'seed': [], 'value': []}
              for label, _ in OUTCOMES}
    names = {}
    for path in paths:
        label = setting_label(path)
        display, code = CONFIG_CODES.get(label, (label, None))
        if code is None:
            print(f'  Skipping unrecognized configuration directory: {label!r}')
            continue
        names[code] = display
        all_results, *_ = load_results(path)
        for alpha, res in zip(ALIGNMENT_SWEEP, all_results):
            for out_label, runs_key in OUTCOMES:
                vals = res.get(runs_key, [])
                for seed, v in enumerate(vals):
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        continue
                    t = tables[out_label]
                    t['alpha'].append(alpha)
                    t['config'].append(code)
                    t['seed'].append(seed)
                    t['value'].append(float(v))
    return tables, names


def fit_outcome(t):
    """Fit the standardized alpha (linear+quadratic) x configuration model.

    Returns (params, bse, pvalues, method, n) as dicts keyed by term name.
    """
    import pandas as pd
    import statsmodels.formula.api as smf

    df = pd.DataFrame({k: np.asarray(v, dtype=float) for k, v in t.items()})
    if df.empty or df['value'].std(ddof=0) < 1e-12:
        return None
    df['z'] = (df['value'] - df['value'].mean()) / df['value'].std(ddof=0)
    df['alpha2'] = df['alpha'] ** 2
    formula = 'z ~ alpha + alpha2 + config + alpha:config + alpha2:config'

    # Primary: mixed model with a random intercept per seed (the seed pairs
    # replications across alpha levels and configurations).
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            md = smf.mixedlm(formula, df, groups=df['seed'])
            fit = md.fit(reml=True, method='lbfgs', maxiter=200)
            if fit.converged:
                terms = [p for p in fit.params.index if p != 'Group Var']
                return ({k: fit.params[k] for k in terms},
                        {k: fit.bse[k] for k in terms},
                        {k: fit.pvalues[k] for k in terms},
                        'MixedLM (random intercept per seed)', len(df))
        except Exception:
            pass
        # Fallback: OLS with seed-clustered (cluster-robust) standard errors.
        fit = smf.ols(formula, df).fit(cov_type='cluster',
                                       cov_kwds={'groups': df['seed']})
        return ({k: fit.params[k] for k in fit.params.index},
                {k: fit.bse[k] for k in fit.bse.index},
                {k: fit.pvalues[k] for k in fit.pvalues.index},
                'OLS, seed-clustered SE (MixedLM fallback)', len(df))


def holm(pvals):
    """Holm-Bonferroni adjustment. Returns adjusted p-values (same order)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def paired_contrasts(t):
    """Per-alpha paired (by seed) contrasts main − control with Holm correction.

    Returns [(alpha, mean_delta, ci_lo, ci_hi, p_holm, n)] or None when the
    two configurations cannot be paired.
    """
    from scipy import stats

    alpha_arr = np.asarray(t['alpha'])
    config_arr = np.asarray(t['config'])
    seed_arr = np.asarray(t['seed'])
    value_arr = np.asarray(t['value'])
    rows, pvals = [], []
    for a in ALIGNMENT_SWEEP:
        d = {}
        for code in (0, 1):
            sel = (alpha_arr == a) & (config_arr == code)
            d[code] = dict(zip(seed_arr[sel], value_arr[sel]))
        shared = sorted(set(d[0]) & set(d[1]))
        if len(shared) < 2:
            continue
        deltas = np.array([d[1][s] - d[0][s] for s in shared])
        n = len(deltas)
        mean = float(deltas.mean())
        se = float(deltas.std(ddof=1) / np.sqrt(n))
        tcrit = stats.t.ppf(0.975, n - 1)
        if se < 1e-15:
            p = 0.0 if abs(mean) > 1e-15 else 1.0
        else:
            p = float(stats.ttest_rel([d[1][s] for s in shared],
                                      [d[0][s] for s in shared]).pvalue)
        rows.append([a, mean, mean - tcrit * se, mean + tcrit * se, p, n])
        pvals.append(p)
    if not rows:
        return None
    adj = holm(pvals)
    for row, p_adj in zip(rows, adj):
        row[4] = float(p_adj)
    return rows


TERM_LABELS = [
    ('Intercept',     'Intercept'),
    ('alpha',         'α (linear)'),
    ('alpha2',        'α² (quadratic)'),
    ('config',        'configuration (main − control)'),
    ('alpha:config',  'α × configuration'),
    ('alpha2:config', 'α² × configuration'),
]


def stars_for(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else '.'
    save_dir = sys.argv[2] if len(sys.argv) > 2 else '.'
    paths = sorted(glob.glob(os.path.join(root, '**', 'experiment_results.json'),
                             recursive=True))
    if not paths:
        print(f'No experiment_results.json found under {root!r}')
        sys.exit(1)
    print('Configurations found: '
          + ', '.join(setting_label(p) for p in paths))

    tables, names = collect_long(paths)
    have_both = len(names) == 2
    os.makedirs(save_dir, exist_ok=True)

    md_path = os.path.join(save_dir, 'regression_table.md')
    csv_path = os.path.join(save_dir, 'regression_table.csv')
    csv_rows = [('outcome', 'term', 'coef_std', 'se', 'p', 'method', 'n')]

    with open(md_path, 'w') as f:
        f.write('# Alignment-sweep regression (M4)\n\n')
        f.write('Late-run (last 75 ticks) per-replicate outcomes. Model per '
                'outcome: z-scored outcome ~ α + α² + configuration + '
                'interactions, replicate seed as grouping factor (MixedLM '
                'random intercept; seed-clustered OLS fallback). '
                'Coefficients are standardized effects (SD units); '
                'configuration is 0 = control (global access), 1 = main '
                'model (network-bounded). '
                '\\*p<0.05, \\*\\*p<0.01, \\*\\*\\*p<0.001.\n\n')

        f.write('## Standardized mixed-model coefficients\n\n')
        header = ['outcome'] + [lbl for _, lbl in TERM_LABELS] + ['method', 'n']
        f.write('| ' + ' | '.join(header) + ' |\n')
        f.write('|' + '---|' * len(header) + '\n')
        for out_label, _ in OUTCOMES:
            t = tables[out_label]
            if not t['value']:
                f.write(f'| {out_label} | ' + 'n/a |' * len(TERM_LABELS)
                        + ' metric absent | 0 |\n')
                continue
            fit = fit_outcome(t)
            if fit is None:
                f.write(f'| {out_label} | ' + 'n/a |' * len(TERM_LABELS)
                        + ' degenerate outcome | 0 |\n')
                continue
            params, bse, pv, method, n = fit
            cells = [out_label]
            for term, _lbl in TERM_LABELS:
                if term in params:
                    cells.append(f'{params[term]:+.3f} ({bse[term]:.3f})'
                                 f'{stars_for(pv[term])}')
                    csv_rows.append((out_label, term, f'{params[term]:.6g}',
                                     f'{bse[term]:.6g}', f'{pv[term]:.3g}',
                                     method, n))
                else:
                    cells.append('—')
            cells += [method, str(n)]
            f.write('| ' + ' | '.join(cells) + ' |\n')

        f.write('\n## Holm-corrected paired per-level contrasts '
                '(main − control)\n\n')
        if not have_both:
            f.write('*Skipped: needs exactly the two paired configurations '
                    '(control + main model).*\n')
        else:
            f.write('Per-seed deltas at each α (replicate *i* shares seed *i* '
                    'across configurations); 95% CI from the paired t '
                    'distribution; p-values Holm-corrected within each '
                    'outcome (11 levels). Bold = adjusted p < 0.05.\n\n')
            for out_label, _ in OUTCOMES:
                t = tables[out_label]
                rows = paired_contrasts(t) if t['value'] else None
                if not rows:
                    f.write(f'### {out_label}\n\n*not pairable / metric '
                            'absent*\n\n')
                    continue
                f.write(f'### {out_label}\n\n')
                f.write('| α | Δ (main − control) | 95% CI | p (Holm) | n |\n')
                f.write('|---|---|---|---|---|\n')
                for a, mean, lo, hi, p, n in rows:
                    mark = '**' if p < 0.05 else ''
                    f.write(f'| {a:.1f} | {mark}{mean:+.3f}{mark} | '
                            f'[{lo:+.3f}, {hi:+.3f}] | {p:.3g} | {n} |\n')
                f.write('\n')

    with open(csv_path, 'w') as f:
        for row in csv_rows:
            f.write(','.join(str(c) for c in row) + '\n')

    print(f'Regression tables saved: {md_path}, {csv_path}')
    with open(md_path) as f:
        print('\n' + f.read())


if __name__ == '__main__':
    main()
