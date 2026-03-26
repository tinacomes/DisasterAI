#!/usr/bin/env python3
"""
Single-condition simulation runner for parallel CI execution.

Runs --n_runs replications for one parameter set and saves results as JSON.
Each GitHub Actions matrix job calls this script once, uploads the JSON,
and the downstream `plot` job aggregates everything.

Usage:
  # Alpha sweep (one job per level):
  python3 simulate.py --outfile results/alpha_0.5.json --alpha 0.5 --n_runs 10

  # Factor sweep (one job per condition):
  python3 simulate.py --outfile results/factor_rumor_1.0.json \
      --alpha 0.5 --rumor_probability 1.0 --n_runs 10
"""

import argparse
import json
import os

import numpy as np
from DisasterAI_Model import DisasterModel, HumanAgent

BASE_PARAMS = {
    'share_exploitative': 0.5,
    'share_of_disaster': 0.15,
    'initial_trust': 0.3,
    'initial_ai_trust': 0.25,
    'number_of_humans': 100,
    'share_confirming': 0.7,
    'disaster_dynamics': 2,
    'width': 30,
    'height': 30,
    'ticks': 100,
    'learning_rate': 0.1,
    'epsilon': 0.3,
    'exploit_trust_lr': 0.015,
    'explor_trust_lr': 0.03,
}


def run_one(params):
    """Run one simulation and return a JSON-serialisable metrics dict."""
    model = DisasterModel(**params)

    seci_exploit, seci_explor                   = [], []
    aeci_exploit, aeci_explor                   = [], []
    mae_exploit,  mae_explor                    = [], []
    prec_exploit, prec_explor                   = [], []
    ai_query_ratio_exploit, ai_query_ratio_explor = [], []
    metric_ticks = []

    # Accumulate relief tokens per 5-tick window for precision calculation
    _win_ex_correct = _win_ex_total = _win_er_correct = _win_er_total = 0
    # Cumulative AI call counters for per-tick delta computation
    _prev_ai_ex = _prev_tot_ex = _prev_ai_er = _prev_tot_er = 0

    for tick in range(params['ticks']):
        model.step()

        # Per-tick AI query ratio (delta of cumulative accum_calls counters)
        ai_ex = tot_ex = ai_er = tot_er = 0
        for _ag in model.agent_list:
            if not isinstance(_ag, HumanAgent):
                continue
            _ai  = getattr(_ag, 'accum_calls_ai',    0)
            _tot = getattr(_ag, 'accum_calls_total',  0)
            if _ag.agent_type == 'exploitative':
                ai_ex += _ai;  tot_ex += _tot
            else:
                ai_er += _ai;  tot_er += _tot
        d_ai_ex,  _prev_ai_ex  = ai_ex  - _prev_ai_ex,  ai_ex
        d_tot_ex, _prev_tot_ex = tot_ex - _prev_tot_ex, tot_ex
        d_ai_er,  _prev_ai_er  = ai_er  - _prev_ai_er,  ai_er
        d_tot_er, _prev_tot_er = tot_er - _prev_tot_er, tot_er
        ai_query_ratio_exploit.append(d_ai_ex / d_tot_ex if d_tot_ex > 0 else float('nan'))
        ai_query_ratio_explor.append( d_ai_er / d_tot_er if d_tot_er > 0 else float('nan'))

        # Accumulate token-based precision counts using current-tick disaster state
        for pos, cnts in model.tokens_this_tick.items():
            is_high = model.disaster_grid[pos] >= 3
            ex_n = cnts.get('exploit', 0)
            er_n = cnts.get('explor', 0)
            if is_high:
                _win_ex_correct += ex_n
                _win_er_correct += er_n
            _win_ex_total += ex_n
            _win_er_total += er_n

        if tick % 5 == 0:
            # SECI and AECI: collect only at the same cadence as MAE/precision so
            # they share the metric_ticks x-axis without repeated values between
            # computation points (the model recomputes both every 5 ticks).
            seci_exploit.append(float(model.seci_data[-1][1]) if model.seci_data else float('nan'))
            seci_explor.append( float(model.seci_data[-1][2]) if model.seci_data else float('nan'))
            aeci_exploit.append(float(model.aeci_data[-1][1]) if model.aeci_data else float('nan'))
            aeci_explor.append( float(model.aeci_data[-1][2]) if model.aeci_data else float('nan'))

            ex_errors, er_errors = [], []
            for agent in model.agent_list:
                if not isinstance(agent, HumanAgent):
                    continue
                # MAE over actual disaster cells only (disaster_grid >= 1).
                # This measures whether the agent FOUND the disaster, not whether
                # they are confident about the majority of non-disaster cells.
                # Filtering by confidence > 0.1 instead gave lower MAE at high α
                # because confirming AI builds confidence on correct L0 (no-disaster)
                # cells, dominating the average with 0-error entries.
                disaster_beliefs = [(c, b) for c, b in agent.beliefs.items()
                                    if isinstance(b, dict) and model.disaster_grid[c] >= 1]
                err = float(np.mean([
                    abs(b.get('level', 0) - model.disaster_grid[c])
                    for c, b in disaster_beliefs
                ])) if disaster_beliefs else float('nan')
                if agent.agent_type == 'exploitative':
                    ex_errors.append(err)
                else:
                    er_errors.append(err)
            mae_exploit.append(float(np.nanmean(ex_errors)) if ex_errors else float('nan'))
            mae_explor.append( float(np.nanmean(er_errors)) if er_errors else float('nan'))
            prec_exploit.append(float(_win_ex_correct / _win_ex_total) if _win_ex_total > 0 else None)
            prec_explor.append( float(_win_er_correct / _win_er_total) if _win_er_total > 0 else None)
            _win_ex_correct = _win_ex_total = _win_er_correct = _win_er_total = 0
            metric_ticks.append(tick)

    return {
        'seci_exploit':           seci_exploit,
        'seci_explor':            seci_explor,
        'aeci_exploit':           aeci_exploit,
        'aeci_explor':            aeci_explor,
        'mae_exploit':            mae_exploit,
        'mae_explor':             mae_explor,
        'prec_exploit':           prec_exploit,
        'prec_explor':            prec_explor,
        'ai_query_ratio_exploit': ai_query_ratio_exploit,
        'ai_query_ratio_explor':  ai_query_ratio_explor,
        'unmet_needs':            [float(v) for v in model.unmet_needs_evolution],
        'metric_ticks':           metric_ticks,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Run one DisasterAI condition N times and save results as JSON.'
    )
    parser.add_argument('--outfile',            required=True,  help='Output JSON path')
    parser.add_argument('--alpha',              type=float, required=True,
                        help='ai_alignment_level (0=truthful, 1=fully confirming)')
    parser.add_argument('--n_runs',             type=int,   default=10,
                        help='Number of independent replications')
    parser.add_argument('--rumor_probability',  type=float, default=0.3,
                        help='Probability each network component receives a rumour')
    parser.add_argument('--disaster_dynamics',  type=int,   default=2,
                        help='Disaster evolution speed: 0=static 1=slow 2=medium 3=rapid')
    parser.add_argument('--share_exploitative', type=float, default=0.5,
                        help='Fraction of agents that are exploitative')
    args = parser.parse_args()

    params = BASE_PARAMS.copy()
    params['ai_alignment_level']  = args.alpha
    params['rumor_probability']   = args.rumor_probability
    params['disaster_dynamics']   = args.disaster_dynamics
    params['share_exploitative']  = args.share_exploitative

    print(f'Condition: α={args.alpha}, rumor={args.rumor_probability}, '
          f'disaster={args.disaster_dynamics}, exploit={args.share_exploitative}')
    print(f'Running {args.n_runs} replications...')

    runs = []
    for i in range(args.n_runs):
        print(f'  Run {i + 1}/{args.n_runs}...')
        runs.append(run_one(params))

    out = {
        'condition': {k: params[k] for k in params},
        'n_runs': args.n_runs,
        'runs': runs,
    }

    outdir = os.path.dirname(os.path.abspath(args.outfile))
    os.makedirs(outdir, exist_ok=True)
    with open(args.outfile, 'w') as f:
        json.dump(out, f)
    print(f'Saved {len(runs)} runs → {args.outfile}')


if __name__ == '__main__':
    main()
