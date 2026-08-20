#!/usr/bin/env python3
"""Summarize robustness-sweep worker JSONs (M5) / salience experiment (M7)
into per-sweep Markdown tables.

Consumes per-condition JSONs written by test_filter_bubbles.py --single-alpha
workers, named by the convention

    robust_<sweep>_<level>_alpha_<a>.json

e.g. robust_pop_300_alpha_0.5.json, robust_net_smallworld_alpha_0.9.json,
robust_ai_10_alpha_0.7.json, robust_verif_0.1_alpha_1.0.json,
robust_salience_0.5_alpha_0.8.json — searched recursively under <root>.

For every (sweep, level) it prints one table row per alpha with the late-run
outcome means (SECI per type, AECI-IE per type, MAE per type, unmet needs,
precision per type, explorer AI trust) and the recovery/transition scalars
(seci_break per type — the 'recovery ticks' context for the verification
sweep; trust_cross for the salience/C12 decision).

The tables are the raw material for docs/robustness_summary.md: one written
paragraph per sweep stating whether the structural precondition, the interior
optimum, and the starvation/capture mechanisms hold (author judgement, not
automated here).

Usage:
    python3 tools/robustness_summary.py <root> [save_dir]

Output: robustness_tables.md (into save_dir, default '.').
"""
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_filter_bubbles import _normalize_result_conventions  # noqa: E402

FNAME_RE = re.compile(r'robust_([A-Za-z]+)_([A-Za-z0-9.]+)_alpha_([0-9.]+)\.json$')

# column label → *_ss_runs key
COLS = [
    ('SECI_ex',    'seci_exploit_ss_runs'),
    ('SECI_er',    'seci_explor_ss_runs'),
    ('AECI-IE_ex', 'aeci_ie_exploit_ss_runs'),
    ('AECI-IE_er', 'aeci_ie_explor_ss_runs'),
    ('AECI-IEc_ex', 'aeci_ie_chan_exploit_ss_runs'),
    ('AECI-IEc_er', 'aeci_ie_chan_explor_ss_runs'),
    ('MAE_ex',     'mae_exploit_ss_runs'),
    ('MAE_er',     'mae_explor_ss_runs'),
    ('Unmet',      'unmet_needs_ss_runs'),
    ('Prec_ex',    'prec_exploit_ss_runs'),
    ('Prec_er',    'prec_explor_ss_runs'),
    ('AItrust_er', 'trust_ai_explor_ss_runs'),
]
# transition scalars: label → *_runs key (per-replication; NaN-safe mean)
SCALARS = [
    ('SECIbreak_ex', 'seci_break_exploit_runs'),
    ('SECIbreak_er', 'seci_break_explor_runs'),
]


def cell(res, key):
    vals = np.asarray(res.get(key, []), dtype=float)
    vals = vals[~np.isnan(vals)] if len(vals) else vals
    if len(vals) == 0:
        return 'n/a'
    m = float(np.mean(vals))
    se = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return f'{m:.3g} (±{se:.2g})'


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else '.'
    save_dir = sys.argv[2] if len(sys.argv) > 2 else '.'

    sweeps = {}   # sweep -> level -> alpha -> result
    for path in sorted(glob.glob(os.path.join(root, '**', 'robust_*.json'),
                                 recursive=True)):
        m = FNAME_RE.search(os.path.basename(path))
        if not m:
            print(f'  Skipping unrecognized file name: {path}')
            continue
        sweep, level, alpha = m.group(1), m.group(2), float(m.group(3))
        with open(path) as f:
            d = _normalize_result_conventions(json.load(f))
        sweeps.setdefault(sweep, {}).setdefault(level, {})[alpha] = d['result']
    if not sweeps:
        print(f'No robust_*.json files found under {root!r}')
        sys.exit(1)

    os.makedirs(save_dir, exist_ok=True)
    md_path = os.path.join(save_dir, 'robustness_tables.md')
    with open(md_path, 'w') as f:
        f.write('# Robustness sweep tables (M5 / M7 raw material)\n\n')
        f.write('Late-run (last 75 ticks) means with across-replication SE. '
                'One table per (sweep, level); the verdict paragraphs '
                '(structural precondition / interior optimum / '
                'starvation-capture mechanisms) go to '
                'docs/robustness_summary.md.\n\n')
        for sweep in sorted(sweeps):
            for level in sorted(sweeps[sweep], key=str):
                f.write(f'## sweep `{sweep}`, level `{level}`\n\n')
                headers = (['α'] + [c for c, _ in COLS]
                           + [s for s, _ in SCALARS])
                f.write('| ' + ' | '.join(headers) + ' |\n')
                f.write('|' + '---|' * len(headers) + '\n')
                for alpha in sorted(sweeps[sweep][level]):
                    res = sweeps[sweep][level][alpha]
                    cells = [f'{alpha:.1f}']
                    cells += [cell(res, key) for _, key in COLS]
                    cells += [cell(res, key) for _, key in SCALARS]
                    f.write('| ' + ' | '.join(cells) + ' |\n')
                f.write('\n')
    print(f'Robustness tables saved: {md_path}')
    with open(md_path) as f:
        print('\n' + f.read())


if __name__ == '__main__':
    main()
