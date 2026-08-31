#!/usr/bin/env python3
"""E1: extract docking-fit targets from Glickman & Sharot's published data.

Source: https://github.com/affective-brain-lab/BiasedHumanAI (public data
and analysis code for Glickman & Sharot, "How human-AI feedback loops alter
human perceptual, emotional and social judgements", Nat. Hum. Behav. 9,
345-359, 2025). Raw data is NOT redistributed here; this script reads a
local clone and writes only derived summary statistics to targets.json,
with the clone's commit hash recorded as provenance.

Conventions mirror the authors' own analysisExp1.m / analysisExp2.m:

Exp1 (emotion aggregation, Level 3 files): 450 trials per subject =
150 solo-baseline trials followed by 6 collaboration blocks of 50.
Bias = P(response 'more sad'); the scale's indifference point is 0.5.
Induced bias = collaboration mean - own solo baseline (per subject).
Conditions used: H-AI (interaction with the biased CNN) and H-H-H
(interaction with another human) -- the two the docking experiment
reproduces. The perceived-as conditions are out of the model's scope.

Exp2 (RDK estimation, Exp2.csv): within-subject conditions
0=baseline (no partner), 1=accurate AI, 2=biased AI, 3=noisy AI.
Bias = response - evidence; error = |response - evidence|.

Fit targets (dimensionless, computed at group level, bootstrap CIs over
subjects). The Level-3 solo baselines sit at ~0.50, so bias induction is
driven entirely by the partner; the natural dimensionless quantity is a
TRANSMISSION coefficient -- the fraction of the partner-self judgment
gap the human adopts:
  kappa = induced_bias / (partner_sad_rate - own_baseline_sad_rate)
computed with induced bias at the final collaboration block (endpoint,
matching the model's end-of-interaction statistic; the mean-over-blocks
variant is reported alongside as robustness):
  T1  kappa_ai:    transmission from the biased-AI partner
  T2  kappa_human: transmission from the human partner
  A   asymmetry T1/T2 (the paper's headline: AI sways more than humans)
  C1  ordering: T1 > T2
  C2  accurate-AI correction (Exp2): error ratio accurate/baseline < 1
  C3  biased-AI bias induction (Exp2): bias(biased) - bias(baseline) > 0

Usage:
  python3 experiments/docking_fit/compute_targets.py \
      --data-dir /path/to/biasedhumanai clone
"""
import argparse
import json
import os
import subprocess

import numpy as np
import pandas as pd

N_BOOT = 10000
RNG = np.random.default_rng(12345)


def exp1_condition_stats(path):
    """Per-subject baseline/collab sad-rates for one Level-3 condition file."""
    df = pd.read_csv(path)
    subjects = sorted(df['subject'].unique())
    rows = []
    for s in subjects:
        d = df[df['subject'] == s].reset_index(drop=True)
        assert len(d) == 450, f'{path}: subject {s} has {len(d)} trials'
        baseline = d['responseSad'][:150].mean()
        blocks = [d['responseSad'][150 + 50 * b:200 + 50 * b].mean()
                  for b in range(6)]
        partner = d['responseAISad'][150:].mean()
        rows.append({'subject': s, 'baseline': baseline, 'partner': partner,
                     **{f'block{b + 1}': v for b, v in enumerate(blocks)}})
    return pd.DataFrame(rows)


def transmission(stats, idx=None, endpoint=True):
    """Group-level kappa = induced bias / (partner - own baseline)."""
    d = stats if idx is None else stats.iloc[idx]
    if endpoint:
        collab = d['block6'].mean()
    else:
        collab = d[[f'block{b}' for b in range(1, 7)]].values.mean()
    base = d['baseline'].mean()
    return (collab - base) / (d['partner'].mean() - base)


def boot_ci(stats, fn):
    n = len(stats)
    vals = np.array([fn(stats, RNG.integers(0, n, n)) for _ in range(N_BOOT)])
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def exp1_targets(data_dir):
    out = {}
    for key, fname in (('human_ai', 'Exp1-Level3-H-AI-H.csv'),
                       ('human_human', 'Exp1-Level3-H-H-H.csv')):
        stats = exp1_condition_stats(
            os.path.join(data_dir, 'Exp1', 'Data', fname))
        collab_cols = [f'block{b}' for b in range(1, 7)]
        induced = stats[collab_cols].mean(axis=1) - stats['baseline']
        per_block = (stats[collab_cols].values
                     - stats['baseline'].values[:, None]).mean(axis=0)
        slope = float(np.polyfit(np.arange(1, 7), per_block, 1)[0])
        out[key] = {
            'file': fname,
            'n_subjects': int(len(stats)),
            'baseline_sad_rate': round(float(stats['baseline'].mean()), 4),
            'collaboration_sad_rate':
                round(float(stats[collab_cols].values.mean()), 4),
            'partner_sad_rate': round(float(stats['partner'].mean()), 4),
            'induced_bias_mean': round(float(induced.mean()), 4),
            'induced_bias_per_block': [round(float(v), 4) for v in per_block],
            'induced_bias_block_slope': round(slope, 5),
            'kappa_endpoint': round(float(transmission(stats)), 3),
            'kappa_endpoint_ci95':
                [round(v, 3) for v in boot_ci(stats, transmission)],
            'kappa_mean_blocks': round(
                float(transmission(stats, endpoint=False)), 3),
            'kappa_mean_blocks_ci95': [round(v, 3) for v in boot_ci(
                stats, lambda s, i: transmission(s, i, endpoint=False))],
        }
    return out


def exp2_targets(data_dir):
    df = pd.read_csv(os.path.join(data_dir, 'Exp2', 'Data', 'Exp2.csv'))
    rows = []
    for s in sorted(df['subject'].unique()):
        d = df[df['subject'] == s]
        r = {'subject': s}
        for cond, name in ((0, 'base'), (1, 'accurate'), (2, 'biased'),
                           (3, 'noisy')):
            dc = d[d['condition'] == cond]
            r[f'bias_{name}'] = (dc['response'] - dc['evidence']).mean()
            r[f'error_{name}'] = (dc['response'] - dc['evidence']).abs().mean()
        rows.append(r)
    st = pd.DataFrame(rows)

    def err_ratio(s, idx=None):
        d = s if idx is None else s.iloc[idx]
        return d['error_accurate'].mean() / d['error_base'].mean()

    def bias_delta(s, idx=None):
        d = s if idx is None else s.iloc[idx]
        return d['bias_biased'].mean() - d['bias_base'].mean()

    return {
        'file': 'Exp2.csv',
        'n_subjects': int(len(st)),
        'bias_baseline': round(float(st['bias_base'].mean()), 3),
        'bias_biased_ai': round(float(st['bias_biased'].mean()), 3),
        'bias_accurate_ai': round(float(st['bias_accurate'].mean()), 3),
        'error_baseline': round(float(st['error_base'].mean()), 3),
        'error_accurate_ai': round(float(st['error_accurate'].mean()), 3),
        'error_biased_ai': round(float(st['error_biased'].mean()), 3),
        'accurate_ai_error_ratio': round(float(err_ratio(st)), 3),
        'accurate_ai_error_ratio_ci95':
            [round(v, 3) for v in boot_ci(st, err_ratio)],
        'biased_ai_induced_bias': round(float(bias_delta(st)), 3),
        'biased_ai_induced_bias_ci95':
            [round(v, 3) for v in boot_ci(st, bias_delta)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True,
                    help='local clone of affective-brain-lab/BiasedHumanAI')
    ap.add_argument('--out', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'targets.json'))
    args = ap.parse_args()

    try:
        commit = subprocess.check_output(
            ['git', '-C', args.data_dir, 'rev-parse', 'HEAD'],
            text=True).strip()
    except Exception:
        commit = 'unknown'

    exp1 = exp1_targets(args.data_dir)
    exp2 = exp2_targets(args.data_dir)

    t1 = exp1['human_ai']['kappa_endpoint']
    t2 = exp1['human_human']['kappa_endpoint']
    targets = {
        'provenance': {
            'source_repo':
                'https://github.com/affective-brain-lab/BiasedHumanAI',
            'source_commit': commit,
            'paper': 'Glickman & Sharot (2025) Nat. Hum. Behav. 9, 345-359, '
                     'doi:10.1038/s41562-024-02077-2',
            'conventions': 'Mirrors analysisExp1.m (450 trials/subject: '
                           '150 solo baseline + 6x50 collaboration blocks; '
                           'bias = P(more sad); induced bias = collaboration '
                           '- own baseline) and analysisExp2.m (condition '
                           '0/1/2/3 = baseline/accurate/biased/noisy AI; '
                           'bias = response - evidence).',
            'note': 'Derived summary statistics only; raw data not '
                    'redistributed.',
        },
        'exp1': exp1,
        'exp2': exp2,
        'fit_targets': {
            'T1_kappa_ai': t1,
            'T1_ci95': exp1['human_ai']['kappa_endpoint_ci95'],
            'T2_kappa_human': t2,
            'T2_ci95': exp1['human_human']['kappa_endpoint_ci95'],
            'A_asymmetry_ai_over_human': round(t1 / t2, 2) if t2 else None,
            'C1_ordering_ai_gt_human': bool(t1 > t2),
            'C2_accurate_ai_error_ratio_lt_1':
                exp2['accurate_ai_error_ratio'],
            'C3_biased_ai_induced_bias_gt_0': exp2['biased_ai_induced_bias'],
        },
    }
    with open(args.out, 'w') as f:
        json.dump(targets, f, indent=2)
    print(json.dumps(targets['fit_targets'], indent=2))
    print(f'\nWrote {args.out}')


if __name__ == '__main__':
    main()
