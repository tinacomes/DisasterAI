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

    seci_exploit, seci_explor = [], []
    aeci_exploit, aeci_explor = [], []
    mae_exploit,  mae_explor  = [], []
    prec_exploit, prec_explor = [], []
    metric_ticks = []

    for tick in range(params['ticks']):
        model.step()

        if model.seci_data:
            s = model.seci_data[-1]
            seci_exploit.append(float(s[1]))
            seci_explor.append(float(s[2]))

        if model.aeci_data:
            a = model.aeci_data[-1]
            aeci_exploit.append(float(a[1]))
            aeci_explor.append(float(a[2]))

        if tick % 5 == 0:
            ex_errors, er_errors = [], []
            ex_correct = ex_total = er_correct = er_total = 0
            for agent in model.agent_list:
                if not isinstance(agent, HumanAgent):
                    continue
                err = float(np.mean([
                    abs(b.get('level', 0) - model.disaster_grid[c])
                    for c, b in agent.beliefs.items()
                    if isinstance(b, dict)
                ])) if agent.beliefs else 0.0
                total = agent.correct_targets + agent.incorrect_targets
                if agent.agent_type == 'exploitative':
                    ex_errors.append(err)
                    ex_correct += agent.correct_targets
                    ex_total   += total
                else:
                    er_errors.append(err)
                    er_correct += agent.correct_targets
                    er_total   += total
            mae_exploit.append(float(np.mean(ex_errors)) if ex_errors else 0.0)
            mae_explor.append( float(np.mean(er_errors)) if er_errors else 0.0)
            prec_exploit.append(float(ex_correct / ex_total) if ex_total > 0 else None)
            prec_explor.append( float(er_correct / er_total) if er_total > 0 else None)
            metric_ticks.append(tick)

    return {
        'seci_exploit': seci_exploit,
        'seci_explor':  seci_explor,
        'aeci_exploit': aeci_exploit,
        'aeci_explor':  aeci_explor,
        'mae_exploit':  mae_exploit,
        'mae_explor':   mae_explor,
        'prec_exploit': prec_exploit,
        'prec_explor':  prec_explor,
        'unmet_needs':  [float(v) for v in model.unmet_needs_evolution],
        'metric_ticks': metric_ticks,
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
