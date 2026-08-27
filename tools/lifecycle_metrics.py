#!/usr/bin/env python3
"""Lifecycle (dynamics) metrics from archived sweep trajectories — PROTOTYPE.

Computes the echo-chamber *dynamics* layer proposed for the PNAS revision
from the cross-replication mean trajectories already archived in
``experiment_results.json`` (e.g. ``results-mechfix/plots-config-*/``):

  * chamber formation tick, peak depth, recovery tick (right-censored at the
    run horizon), and persistence fraction — per agent type and population,
    per alpha, per configuration (thresholds identical to
    ``test_filter_bubbles.py``: formation SECI < -0.1, recovery sustained
    above -0.05);
  * capture onset: first sustained tick with the exploiters' AI query share
    >= 0.5;
  * mechanism-cascade half-crossing ticks (capture -> L1+ pool collapse ->
    explorer chamber -> lock-in -> precision decline -> periphery aid gap),
    i.e. the tick at which each mechanism variable first sustainedly crosses
    halfway from its initial value to its steady-state extreme;
  * harm-accumulation slopes over the final 50 ticks (stationarity check:
    a non-zero slope means the outcome had NOT equilibrated at the horizon).

PROTOTYPE STATUS: everything here is computed from the *mean* trajectory
across replications, because the archived JSONs store per-seed values only
at steady state. The numbers are suitable for drafting text and figure
design, NOT for the paper's statistical claims — those need per-seed
trajectories (re-dispatch the primary sweep with per-seed lifecycle
columns; metrics are observation-only, so all existing series reproduce
bit-identically on the same seeds).

Usage:
  python3 tools/lifecycle_metrics.py \
      --switches results-mechfix/plots-config-switches/experiment_results.json \
      --baseline results-mechfix/plots-config-baseline/experiment_results.json \
      --outdir   results-mechfix/lifecycle
"""

import argparse
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

# Thresholds — keep identical to test_filter_bubbles.py so the prototype
# numbers are continuous with the archived lifecycle/transition figures.
FORM_THRESH  = -0.1    # SECI below this = chamber formed
BREAK_THRESH = -0.05   # sustained recovery above this = chamber dissolved
SUSTAIN_5T   = 5       # sustain window, in samples, for 5-tick-cadence series
SUSTAIN_1T   = 5       # sustain window, in ticks, for per-tick series
CAPTURE_THRESH = 0.5   # AI query share majority
CAPTURE_SUSTAIN = 10   # ticks the AI-majority must persist to count as onset
SS_SAMPLES   = 15      # steady-state window: last 15 samples = 75 ticks
SLOPE_TICKS  = 50      # window for the final-slope stationarity check


# --- threshold helpers (logic mirrored from test_filter_bubbles.py; kept
# --- local so this tool runs without importing the simulation stack) -------

def _first_sustained_break(series, form_thresh=FORM_THRESH,
                           break_thresh=BREAK_THRESH, sustain=SUSTAIN_5T):
    """Index of first sustained recovery after formation.

    Returns -1 if no chamber ever formed, len(series) if formed but never
    recovered (right-censored), else the recovery index.
    """
    formed = False
    for i, v in enumerate(series):
        if np.isnan(v):
            continue
        if not formed and v < form_thresh:
            formed = True
        elif formed:
            window = [w for w in series[i:i + sustain] if not np.isnan(w)]
            if len(window) == sustain and all(w > break_thresh for w in window):
                return i
    return -1 if not formed else len(series)


def _first_sustained_cross(series, threshold, sustain, direction='up'):
    """Index of first sustained threshold crossing; len(series) if never."""
    for i, v in enumerate(series):
        if np.isnan(v):
            continue
        ok = (v >= threshold) if direction == 'up' else (v <= threshold)
        if ok:
            window = [w for w in series[i:i + sustain] if not np.isnan(w)]
            good = all((w >= threshold) if direction == 'up' else (w <= threshold)
                       for w in window)
            if len(window) == sustain and good:
                return i
    return len(series)


def _arr(x):
    """Series -> float array with None/NaN normalised."""
    return np.array([float('nan') if v is None else float(v) for v in x],
                    dtype=float)


def _smooth(a, w=5):
    """Centered rolling mean, NaN-safe, same length."""
    out = np.full(len(a), np.nan)
    half = w // 2
    for i in range(len(a)):
        seg = a[max(0, i - half):i + half + 1]
        seg = seg[~np.isnan(seg)]
        if len(seg):
            out[i] = seg.mean()
    return out


def chamber_episodes(seci, metric_ticks, horizon):
    """All chamber episodes of one SECI series, with hysteresis.

    An episode opens when SECI drops below FORM_THRESH and closes at the
    first index from which the series stays above BREAK_THRESH for
    SUSTAIN_5T consecutive samples (same hysteresis as
    ``_first_sustained_break``). Returns a list of (start_tick, end_tick,
    censored) tuples; a censored episode was still open at the horizon.
    """
    a = _arr(seci)
    episodes, start = [], None
    i = 0
    while i < len(a):
        v = a[i]
        if np.isnan(v):
            i += 1
            continue
        if start is None:
            if v < FORM_THRESH:
                start = metric_ticks[i]
        else:
            window = [w for w in a[i:i + SUSTAIN_5T] if not np.isnan(w)]
            if (len(window) == SUSTAIN_5T
                    and all(w > BREAK_THRESH for w in window)):
                episodes.append((start, metric_ticks[i], False))
                start = None
        i += 1
    if start is not None:
        episodes.append((start, horizon, True))
    return episodes


def chamber_lifecycle(seci, metric_ticks, horizon):
    """Formation/peak/recovery/persistence stats for one mean SECI series.

    Chambers can be multi-episode (form, dissolve mid-run, re-form), so the
    first recovery alone can badly misrepresent the endpoint: the stats
    include the episode count, the final dissolution tick (end of the LAST
    episode; censored at the horizon if it never closes) and whether the
    chamber is standing in the steady-state window.
    """
    a = _arr(seci)
    eps = chamber_episodes(seci, metric_ticks, horizon)
    any_valid = np.any(~np.isnan(a))
    ss_n = min(SS_SAMPLES, len(a))
    stats = {
        'formed': bool(eps),
        'formation_tick': eps[0][0] if eps else None,
        'peak_depth': float(np.nanmin(a)) if any_valid else None,
        'peak_tick': metric_ticks[int(np.nanargmin(a))] if any_valid else None,
        'persistence_frac': float(np.nanmean(a < FORM_THRESH)) if any_valid else None,
        'n_episodes': len(eps),
        'episodes': eps,
        'in_chamber_at_end': (float(np.nanmean(a[-ss_n:])) < FORM_THRESH
                              if any_valid else None),
    }
    if not eps:
        stats['first_recovery_tick'] = None
        stats['final_dissolution_tick'], stats['dissolved'] = None, None
    else:
        stats['first_recovery_tick'] = None if eps[0][2] else eps[0][1]
        last = eps[-1]
        stats['final_dissolution_tick'] = last[1]
        stats['dissolved'] = not last[2]
    return stats


def lifecycle_scalars_for_run(run):
    """Per-seed lifecycle scalars from one run's trajectory dict.

    Called by ``test_filter_bubbles._aggregate`` on each replicate while the
    per-run series still exist, so the sweep output carries per-seed
    lifecycle columns (``lc_*_runs``) next to the ``*_ss_runs`` scalars —
    the basis for CIs on the formation/dissolution claims. ``run`` uses the
    per-run key names of ``run_one_sim`` (no ``_mean`` suffix). Metrics are
    observation-only: adding these columns changes no simulated series.

    Returns a flat dict; ticks are ints, flags 1/0, missing values None.
    """
    ticks = run['metric_ticks']
    horizon = len(run.get('unmet_needs', [])) or (ticks[-1] + 5)
    out = {}
    for typ in ('exploit', 'explor', 'pop'):
        key = f'seci_{typ}' if typ != 'pop' else 'seci_pop'
        lc = chamber_lifecycle(run[key], ticks, horizon)
        out[f'lc_{typ}_formation'] = lc['formation_tick']
        out[f'lc_{typ}_peak_depth'] = lc['peak_depth']
        out[f'lc_{typ}_n_episodes'] = lc['n_episodes']
        out[f'lc_{typ}_final_dissolution'] = lc['final_dissolution_tick']
        out[f'lc_{typ}_dissolved'] = (None if lc['dissolved'] is None
                                      else int(lc['dissolved']))
        out[f'lc_{typ}_persistence'] = lc['persistence_frac']
        out[f'lc_{typ}_end_in_chamber'] = (None if lc['in_chamber_at_end'] is None
                                           else int(lc['in_chamber_at_end']))
    for typ in ('exploit', 'explor'):
        share = _smooth(_arr(run[f'ai_query_ratio_{typ}']))
        idx = _first_sustained_cross(list(share), CAPTURE_THRESH,
                                     CAPTURE_SUSTAIN)
        out[f'lc_capture_onset_{typ}'] = idx if idx < len(share) else None
    return out


def half_crossing(series, ticks, sustain):
    """Tick at which the series first sustainedly crosses halfway from its
    initial value to its steady-state (last SS window) value; None when the
    total change is negligible relative to the series' scale."""
    a = _arr(series)
    valid = a[~np.isnan(a)]
    if len(valid) < sustain + 2:
        return None
    ss_n = SS_SAMPLES if len(a) >= 2 * SS_SAMPLES else max(2, len(a) // 4)
    first_valid = np.flatnonzero(~np.isnan(a))[:2]
    base  = float(np.mean(a[first_valid]))
    final = float(np.nanmean(a[-ss_n:]))
    change = final - base
    scale = max(np.nanmax(a) - np.nanmin(a), 1e-9)
    if abs(change) < 0.25 * scale or abs(change) < 1e-6:
        return None                       # no meaningful net transition
    thresh = base + 0.5 * change
    idx = _first_sustained_cross(list(a), thresh, sustain,
                                 'up' if change > 0 else 'down')
    return ticks[idx] if idx < len(a) else None


def final_slope(series, ticks, window_ticks=SLOPE_TICKS):
    """OLS slope (per 100 ticks) over the final window; NaN-safe."""
    a, t = _arr(series), np.asarray(ticks, dtype=float)
    m = t >= (t[-1] - window_ticks)
    a, t = a[m], t[m]
    keep = ~np.isnan(a)
    if keep.sum() < 3:
        return None
    b = np.polyfit(t[keep], a[keep], 1)[0]
    return float(b * 100.0)


# --------------------------------------------------------------------------

CASCADE_VARS = [
    # (label, key, per-tick?, smooth?). AECI-LockIn is deliberately absent:
    # its mean trajectory has a large initialisation transient that defeats
    # half-crossing timing at this resolution — freeze timing needs the
    # per-seed re-run.
    ('AI capture (exploit query share)', 'ai_query_ratio_exploit_mean', True,  True),
    ('Belief starvation (explor L1+ pool)', 'l1pool_explor_mean',       False, False),
    ('Explorer chamber (SECI)',          'seci_explor_mean',            False, False),
    ('Precision decline (explor)',       'prec_explor_mean',            False, False),
    ('Periphery aid gap (spatial)',      'periph_sp_aid_gap_mean',      False, False),
]

SLOPE_VARS = [
    ('periph_sp_mae_gap_mean', False),
    ('periph_sp_aid_gap_mean', False),
    ('unmet_needs_mean',       True),
]


def analyse_config(data, label):
    """Compute all lifecycle rows for one configuration's experiment JSON."""
    alphas  = data['alignment_sweep']
    rows = []
    for alpha, res in zip(alphas, data['all_results']):
        ticks   = res['metric_ticks']
        horizon = res.get('n_ticks', ticks[-1] + 5)
        row = {'config': label, 'alpha': alpha}
        for typ, key in (('exploit', 'seci_exploit_mean'),
                         ('explor',  'seci_explor_mean'),
                         ('pop',     'seci_pop_mean')):
            lc = chamber_lifecycle(res[key], ticks, horizon)
            lc['episodes'] = '; '.join(
                f"{s}-{e}{'+' if c else ''}" for s, e, c in lc['episodes'])
            for k, v in lc.items():
                row[f'{typ}_{k}'] = v
        # Capture onset from the per-tick query-share series (smoothed).
        for typ in ('exploit', 'explor'):
            share = _smooth(_arr(res[f'ai_query_ratio_{typ}_mean']))
            idx = _first_sustained_cross(list(share), CAPTURE_THRESH,
                                         CAPTURE_SUSTAIN)
            row[f'{typ}_capture_onset'] = idx if idx < len(share) else None
        # Mechanism-cascade half-crossings.
        for lab, key, per_tick, smooth in CASCADE_VARS:
            series = res[key]
            t = list(range(len(series))) if per_tick else ticks
            s = _smooth(_arr(series)) if smooth else series
            row[f'halfx_{key}'] = half_crossing(
                s, t, SUSTAIN_1T if per_tick else 3)
        # Final-window slopes (per 100 ticks).
        for key, per_tick in SLOPE_VARS:
            t = list(range(len(res[key]))) if per_tick else ticks
            row[f'slope_{key}'] = final_slope(res[key], t)
        rows.append(row)
    return rows


# ------------------------------ output ------------------------------------

def _mean_ci95(vals):
    """(mean, 95% CI half-width) over non-None values; (None, None) if empty."""
    a = np.array([v for v in vals if v is not None], dtype=float)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return None, None
    hw = 1.96 * a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else float('nan')
    return float(a.mean()), float(hw)


def write_perseed_tables(datasets, outdir):
    """CI-grade per-seed lifecycle tables from the lc_*_runs columns.

    Only possible on sweeps run with the instrumented ``_aggregate`` (the
    per-seed lifecycle re-run); returns None when the archive predates the
    columns, in which case only the mean-trajectory prototype tables exist.
    """
    probe = datasets[next(iter(datasets))]['all_results'][0]
    if 'lc_explor_formation_runs' not in probe:
        return None

    md_path = os.path.join(outdir, 'lifecycle_perseed.md')
    with open(md_path, 'w') as f:
        f.write('# Echo-chamber lifecycle — PER-SEED statistics (CI-grade)\n\n')
        f.write('Computed from the per-seed lifecycle columns (`lc_*_runs`) '
                'of the instrumented sweep: each replicate\'s own trajectory '
                'is classified before aggregation, so fractions are seed '
                'counts and intervals are across-replication 95% CIs. '
                'Thresholds as in the prototype layer (formation SECI '
                f'< {FORM_THRESH}, dissolution sustained > {BREAK_THRESH}).\n\n')
        for typ, name in (('exploit', 'confirmation-seeking (exploitative)'),
                          ('explor', 'accuracy-seeking (exploratory)'),
                          ('pop', 'population (societal)')):
            f.write(f'## {name} communities\n\n')
            f.write('| config | alpha | formed | formation tick | peak SECI '
                    '| dissolved by end | in chamber at end | persistence |\n')
            f.write('|---|---|---|---|---|---|---|---|\n')
            for label, data in datasets.items():
                for alpha, res in zip(data['alignment_sweep'],
                                      data['all_results']):
                    form = res[f'lc_{typ}_formation_runs']
                    n = len(form)
                    formed = [v for v in form if v is not None]
                    fm, fh = _mean_ci95(formed)
                    pm, ph = _mean_ci95(res[f'lc_{typ}_peak_depth_runs'])
                    dis = [v for v in res[f'lc_{typ}_dissolved_runs']
                           if v is not None]
                    endch = [v for v in res[f'lc_{typ}_end_in_chamber_runs']
                             if v is not None]
                    sm, sh = _mean_ci95(res[f'lc_{typ}_persistence_runs'])
                    f.write(
                        f'| {label} | {alpha} | {len(formed)}/{n} | '
                        + (f'{fm:.0f} ± {fh:.0f}' if fm is not None else '--')
                        + ' | '
                        + (f'{pm:.2f} ± {ph:.2f}' if pm is not None else '--')
                        + f' | {sum(dis)}/{len(dis)}'
                        + f' | {sum(endch)}/{len(endch)}'
                        + ' | '
                        + (f'{sm:.2f} ± {sh:.2f}' if sm is not None else '--')
                        + ' |\n')
            f.write('\n')
        f.write('## Capture onset (first sustained AI-majority tick)\n\n')
        f.write('| config | alpha | onset exploit (mean ± CI, n reached) '
                '| onset explor |\n')
        f.write('|---|---|---|---|\n')
        for label, data in datasets.items():
            for alpha, res in zip(data['alignment_sweep'],
                                  data['all_results']):
                cells = []
                for typ in ('exploit', 'explor'):
                    vals = [v for v in res[f'lc_capture_onset_{typ}_runs']
                            if v is not None]
                    m, h = _mean_ci95(vals)
                    n = len(res[f'lc_capture_onset_{typ}_runs'])
                    cells.append(f'{m:.0f} ± {h:.0f} ({len(vals)}/{n})'
                                 if m is not None else f'-- (0/{n})')
                f.write(f'| {label} | {alpha} | {cells[0]} | {cells[1]} |\n')
    return md_path


def write_tables(rows, outdir):
    keys = list(rows[0].keys())
    csv_path = os.path.join(outdir, 'lifecycle_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    def fmt(v, nd=2):
        if v is None:
            return '--'
        if isinstance(v, bool):
            return 'yes' if v else 'no'
        if isinstance(v, float):
            return f'{v:.{nd}f}'
        return str(v)

    md_path = os.path.join(outdir, 'lifecycle_metrics.md')
    with open(md_path, 'w') as f:
        f.write('# Echo-chamber lifecycle metrics (PROTOTYPE)\n\n')
        f.write('Computed from the cross-replication **mean** trajectories in '
                'the archived `experiment_results.json` files; thresholds '
                f'follow `test_filter_bubbles.py` (formation SECI < {FORM_THRESH}, '
                f'dissolution sustained > {BREAK_THRESH} for {SUSTAIN_5T} samples). '
                'Chambers can be multi-episode (form, dissolve mid-run, '
                're-form), so each row lists every episode; an episode ending '
                'at the run horizon with a `+` was still open when the run '
                'ended (right-censored), and `dissolved? = no` means the LAST '
                'episode never closed. **No CIs: per-seed trajectories are '
                'not archived** — publication statistics need the re-run with '
                'per-seed lifecycle columns.\n\n')
        def _lc_cols(typ):
            return [(f'{typ}_formation_tick', 'formation'),
                    (f'{typ}_peak_depth', 'peak SECI'),
                    (f'{typ}_episodes', 'episodes (start-end, + = censored)'),
                    (f'{typ}_final_dissolution_tick', 'final dissolution'),
                    (f'{typ}_dissolved', 'dissolved?'),
                    (f'{typ}_in_chamber_at_end', 'standing at end?'),
                    (f'{typ}_persistence_frac', 'persistence')]

        for section, cols in (
            ('Chamber lifecycle — confirmation-seeking (exploitative) communities',
             _lc_cols('exploit')),
            ('Chamber lifecycle — accuracy-seeking (exploratory) communities',
             _lc_cols('explor')),
            ('Chamber lifecycle — population (societal) index',
             _lc_cols('pop')),
            ('Capture onset (first sustained AI-majority tick) and final-window '
             'slopes (per 100 ticks; non-zero = not equilibrated at horizon)',
             [('exploit_capture_onset', 'capture onset (exploit)'),
              ('explor_capture_onset', 'capture onset (explor)'),
              ('slope_periph_sp_mae_gap_mean', 'MAE-gap slope'),
              ('slope_periph_sp_aid_gap_mean', 'aid-gap slope'),
              ('slope_unmet_needs_mean', 'unmet slope')]),
            ('Mechanism cascade — half-crossing tick per mechanism '
             '(-- = no meaningful net transition at that alpha)',
             [(f'halfx_{k}', lab) for lab, k, _, _ in CASCADE_VARS]),
        ):
            f.write(f'## {section}\n\n')
            f.write('| config | alpha | ' + ' | '.join(c[1] for c in cols) + ' |\n')
            f.write('|---' * (len(cols) + 2) + '|\n')
            for r in rows:
                f.write(f"| {r['config']} | {r['alpha']} | "
                        + ' | '.join(fmt(r[c[0]]) for c in cols) + ' |\n')
            f.write('\n')
    return csv_path, md_path


# ------------------------------ figures -----------------------------------

def _alpha_color(alpha):
    return plt.cm.viridis(alpha)


def plot_timeline(datasets, outdir):
    """Chamber lifespan bars: x = tick, one bar per alpha, per type/config."""
    fig, axes = plt.subplots(len(datasets), 2, figsize=(12, 4 * len(datasets)),
                             squeeze=False)
    for r, (label, data) in enumerate(datasets.items()):
        alphas = data['alignment_sweep']
        for c, (typ, name) in enumerate((('exploit', 'confirmation-seeking'),
                                         ('explor', 'accuracy-seeking'))):
            ax = axes[r][c]
            for i, (alpha, res) in enumerate(zip(alphas, data['all_results'])):
                ticks   = res['metric_ticks']
                horizon = res.get('n_ticks', ticks[-1] + 5)
                eps = chamber_episodes(res[f'seci_{typ}_mean'], ticks, horizon)
                if not eps:
                    ax.plot(0, i, marker='o', mfc='none', color='grey', ms=5)
                    continue
                for start, end, censored in eps:
                    ax.barh(i, end - start, left=start, height=0.62,
                            color=_alpha_color(alpha), edgecolor='none')
                    if censored:
                        ax.plot(horizon, i, marker='x', color='crimson',
                                ms=8, mew=2, clip_on=False)
            ax.set_yticks(range(len(alphas)))
            ax.set_yticklabels([f'{a:.1f}' for a in alphas])
            ax.set_ylabel('AI alignment α' if c == 0 else '')
            ax.set_xlabel('Simulation tick')
            ax.set_xlim(0, None)
            ax.set_title(f'{label}: {name} chamber lifespan\n'
                         '(bar = chamber standing; red x = never dissolved)')
            ax.invert_yaxis()
    fig.suptitle('PROTOTYPE (mean-trajectory basis) — echo-chamber lifespans:'
                 ' formation is near-universal; alignment blocks dissolution',
                 y=1.001)
    fig.tight_layout()
    path = os.path.join(outdir, 'proto_lifecycle_timeline.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_depth_vs_persistence(datasets, outdir):
    """The money contrast: peak depth ~flat in alpha, dissolution collapses."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    styles = {'main': '-', 'control': '--'}
    for label, data in datasets.items():
        alphas = data['alignment_sweep']
        stats = {typ: [chamber_lifecycle(res[f'seci_{typ}_mean'],
                                         res['metric_ticks'],
                                         res.get('n_ticks', 200))
                       for res in data['all_results']]
                 for typ in ('exploit', 'explor')}
        ls = styles.get(label, '-')
        for typ, color, name in (('exploit', '#1f4e79', 'confirmation-seeking'),
                                 ('explor', '#74b3e3', 'accuracy-seeking')):
            axes[0].plot(alphas, [abs(s['peak_depth']) for s in stats[typ]],
                         ls, color=color, marker='o', ms=4,
                         label=f'{name} ({label})')
            axes[1].plot(alphas,
                         [s['final_dissolution_tick'] if s['formed'] else np.nan
                          for s in stats[typ]],
                         ls, color=color, marker='o', ms=4)
            cens = [(a, s['final_dissolution_tick'])
                    for a, s in zip(alphas, stats[typ])
                    if s['dissolved'] is False]
            if cens:
                axes[1].plot([c[0] for c in cens], [c[1] for c in cens],
                             'x', color='crimson', ms=9, mew=2)
            axes[2].plot(alphas, [s['persistence_frac'] for s in stats[typ]],
                         ls, color=color, marker='o', ms=4)
    axes[0].set_title('Peak chamber depth |SECI|$_{max}$\n(nearly flat in α)')
    axes[0].set_ylabel('peak |SECI|')
    axes[1].set_title('Tick of final chamber dissolution\n(red x = censored: never dissolved)')
    axes[1].set_ylabel('final dissolution tick')
    axes[2].set_title('Fraction of run inside chamber')
    axes[2].set_ylabel('persistence fraction')
    for ax in axes:
        ax.set_xlabel('AI alignment α')
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle('PROTOTYPE (mean-trajectory basis) — alignment does not deepen '
                 'echo chambers; it prevents them from dissolving')
    fig.tight_layout()
    path = os.path.join(outdir, 'proto_depth_vs_persistence.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_trajectories(datasets, outdir, show_alphas=(0.0, 0.6, 1.0)):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for label, data in datasets.items():
        ls = '-' if label == 'main' else '--'
        for alpha, res in zip(data['alignment_sweep'], data['all_results']):
            if round(alpha, 1) not in show_alphas:
                continue
            for ax, typ in zip(axes, ('exploit', 'explor')):
                ax.plot(res['metric_ticks'], _arr(res[f'seci_{typ}_mean']),
                        ls, color=_alpha_color(alpha), lw=1.8,
                        label=f'α={alpha:.1f} ({label})')
    for ax, name in zip(axes, ('confirmation-seeking (exploitative)',
                               'accuracy-seeking (exploratory)')):
        ax.axhline(FORM_THRESH, color='grey', ls=':', lw=1)
        ax.text(2, FORM_THRESH + 0.01, 'formation threshold', fontsize=7,
                color='grey')
        ax.set_title(f'{name} communities')
        ax.set_xlabel('Simulation tick')
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('SECI (negative = echo chamber)')
    axes[1].legend(fontsize=7, ncol=2)
    fig.suptitle('PROTOTYPE — SECI trajectories: low-α chambers dissolve, '
                 'high-α chambers persist (solid = main, dashed = control)')
    fig.tight_layout()
    path = os.path.join(outdir, 'proto_trajectories.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_cascade(rows, outdir, config='main', alphas=(0.8, 1.0)):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    labels = [lab for lab, *_ in CASCADE_VARS]
    keys   = [f'halfx_{k}' for _, k, _, _ in CASCADE_VARS]
    offs   = np.linspace(-0.15, 0.15, len(alphas))
    for off, alpha in zip(offs, alphas):
        row = next(r for r in rows
                   if r['config'] == config and abs(r['alpha'] - alpha) < 1e-9)
        xs = [row[k] for k in keys]
        ys = np.arange(len(labels)) + off
        ok = [(x, y) for x, y in zip(xs, ys) if x is not None]
        if ok:
            ax.plot([x for x, _ in ok], [y for _, y in ok], '-o',
                    color=_alpha_color(alpha), ms=7, label=f'α={alpha:.1f}')
        for x, y in zip(xs, ys):
            if x is None:
                ax.plot(2, y, marker='o', mfc='none', color='lightgrey', ms=5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Tick of half-transition (initial → steady-state extreme)')
    ax.set_title(f'PROTOTYPE — mechanism cascade, {config} model\n'
                 'capture leads; starvation and the chamber co-develop; '
                 'operational harms trail '
                 '(open circle at left = no meaningful transition)')
    ax.grid(alpha=0.3, axis='x')
    ax.legend()
    fig.tight_layout()
    path = os.path.join(outdir, 'proto_mechanism_cascade.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--switches', required=True,
                   help='experiment_results.json of the main model (switches)')
    p.add_argument('--baseline', required=True,
                   help='experiment_results.json of the control (baseline)')
    p.add_argument('--outdir', required=True)
    args = p.parse_args()

    datasets = {}
    for label, path in (('main', args.switches), ('control', args.baseline)):
        with open(path) as f:
            datasets[label] = json.load(f)

    os.makedirs(args.outdir, exist_ok=True)
    rows = []
    for label, data in datasets.items():
        rows.extend(analyse_config(data, label))

    outputs = list(write_tables(rows, args.outdir))
    perseed = write_perseed_tables(datasets, args.outdir)
    if perseed:
        outputs.append(perseed)
    else:
        print('note: no lc_*_runs columns in these archives - per-seed '
              'tables skipped (prototype mean-trajectory outputs only)')
    outputs.append(plot_timeline(datasets, args.outdir))
    outputs.append(plot_depth_vs_persistence(datasets, args.outdir))
    outputs.append(plot_trajectories(datasets, args.outdir))
    outputs.append(plot_cascade(rows, args.outdir))
    for o in outputs:
        print('wrote', o)


if __name__ == '__main__':
    main()
