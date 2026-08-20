#!/usr/bin/env python3
"""M6 dyadic docking: one human x one AI, no network, no relief loop.

Docking target: the dyadic human-AI feedback-loop amplification of Glickman &
Sharot (Nat. Hum. Behav. 2024) — interacting with an AI system that is trained
on / responsive to the human's own judgments amplifies the human's initial
bias more than interacting with another human, and the human-AI loop is the
more biased the more the AI aligns to the human.

Setup (reuses the model's own mechanism code — acceptance windows D/delta,
Bayesian belief revision, and the AI response rule r = (1-alpha)*t + alpha*b
with b = the caller's own current belief — without the population loop):

  * A DisasterModel is built only as a container (fixed epicenter, no rumor
    seeding); the model is NEVER stepped, so there is no relief allocation, no
    movement, no trust learning, and no social network influence.
  * A focal human of each cognitive type receives an implanted biased prior:
    a false epicenter far from the true disaster (the dyadic analogue of the
    rumor mechanism).
  * Human-AI condition: each round the focal human selects an interest point
    with its OWN type-specific policy (exploiters: believed epicenter;
    explorers: highest-uncertainty area), queries the AI, and applies the
    standard D/delta + Bayesian update. The AI conditions on the caller's
    current beliefs — that closed loop IS the dyadic feedback.
  * Human-human condition: the focal human exchanges reports with a same-type
    partner without the implanted bias (bidirectional query-and-update, same
    acceptance machinery) — the reference interaction.

Outcomes per round:
  false_belief  mean believed level over the implanted false-epicenter cells
                (ground truth there is 0) — the BIAS the loop can amplify
  mae           belief MAE over true disaster cells (headline construct)

Docking succeeds qualitatively if (a) the final false-belief bias increases
with alpha in the human-AI dyad and (b) the aligned-AI dyad (high alpha)
retains more of the initial bias than the human-human pair — the Glickman &
Sharot amplification asymmetry. The truthful-AI endpoint is reported
descriptively: for exploitative focal humans even alpha = 0 barely corrects
the bias, because the D/delta acceptance window rejects strongly
disconfirming reports — the model's own acceptance mechanism, not a docking
failure.

Usage:
  python3 experiments/dyadic_docking.py --n-seeds 20 --rounds 40 \
      --save-dir dyadic_results
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DisasterAI_Model import DisasterModel, HumanAgent  # noqa: E402

ALPHAS = [round(0.1 * i, 1) for i in range(11)]

BASE = dict(
    share_exploitative=0.5, share_of_disaster=0.15, initial_trust=0.3,
    initial_ai_trust=0.25, number_of_humans=10, share_confirming=0.7,
    disaster_dynamics=0,          # static hazard — isolates the belief loop
    width=30, height=30, ticks=1, learning_rate=0.1, epsilon=0.3,
    exploit_trust_lr=0.015, explor_trust_lr=0.03,
    rumor_probability=0.0,        # bias is implanted manually (see below)
    epicenter=[22, 22],           # fixed true epicenter, far from the bias
    mobility=0, network_type='components', query_scope='global',
)

FALSE_EPICENTER = (7, 7)          # implanted bias location (true level 0)
FALSE_RADIUS = 3
FALSE_LEVEL = 4
FALSE_CONF = 0.6
QUERY_RADIUS = 2                  # same radius seek_information uses


def build_container(seed):
    """Fresh model container with a fixed seed; never stepped."""
    random.seed(seed)
    np.random.seed(seed)
    return DisasterModel(**BASE)


def implant_bias(agent):
    """Give the agent the false-epicenter prior (rumor-mechanism analogue)."""
    fx, fy = FALSE_EPICENTER
    for dx in range(-FALSE_RADIUS, FALSE_RADIUS + 1):
        for dy in range(-FALSE_RADIUS, FALSE_RADIUS + 1):
            cell = (fx + dx, fy + dy)
            if cell not in agent.beliefs:
                continue
            dist = np.sqrt(dx * dx + dy * dy)
            if dist > FALSE_RADIUS:
                continue
            level = max(1, int(round(FALSE_LEVEL * (1 - dist / (FALSE_RADIUS + 1)))))
            agent.beliefs[cell] = {'level': level, 'confidence': FALSE_CONF}


def false_belief_cells(model):
    fx, fy = FALSE_EPICENTER
    return [(fx + dx, fy + dy)
            for dx in range(-FALSE_RADIUS, FALSE_RADIUS + 1)
            for dy in range(-FALSE_RADIUS, FALSE_RADIUS + 1)
            if np.sqrt(dx * dx + dy * dy) <= FALSE_RADIUS
            and 0 <= fx + dx < model.width and 0 <= fy + dy < model.height]


def measure(agent, model, bias_cells):
    fb = float(np.mean([agent.beliefs.get(c, {'level': 0}).get('level', 0)
                        for c in bias_cells]))
    disaster_cells = [(c, b) for c, b in agent.beliefs.items()
                      if isinstance(b, dict) and model.disaster_grid[c] >= 1]
    mae = float(np.mean([abs(b.get('level', 0) - model.disaster_grid[c])
                         for c, b in disaster_cells])) if disaster_cells else float('nan')
    return fb, mae


def pick_interest_point(agent, model):
    """The agent's own type-specific query policy (mirrors seek_information)."""
    if agent.agent_type == 'exploitative':
        agent.find_believed_epicenter()
        ip = agent.believed_epicenter
    else:
        ip = agent.find_highest_uncertainty_area()
    if ip is None or not (0 <= ip[0] < model.width and 0 <= ip[1] < model.height):
        ip = (random.randrange(model.width), random.randrange(model.height))
    return ip


def apply_report(agent, reports, source_trust, source_id):
    for cell, val in reports.items():
        if cell in agent.beliefs:
            agent.update_belief_bayesian(cell, int(round(val)), source_trust,
                                         source_id)


def run_dyad(seed, agent_type, partner, alpha, rounds):
    """One dyadic run. partner: 'ai' or 'human'. Returns per-round series."""
    params_alpha = alpha if partner == 'ai' else 0.0
    model = build_container(seed)
    model.ai_alignment_level = params_alpha

    humans = [a for a in model.agent_list if isinstance(a, HumanAgent)]
    focal = next(a for a in humans if a.agent_type == agent_type)
    implant_bias(focal)
    bias_cells = false_belief_cells(model)

    ai = next(iter(model.ais.values()))
    mate = next(a for a in humans
                if a.agent_type == agent_type and a is not focal)

    ai_trust = focal.trust.get(ai.unique_id, BASE['initial_ai_trust'])
    hh_trust = focal.trust.get(mate.unique_id, BASE['initial_trust'])

    fb0, mae0 = measure(focal, model, bias_cells)
    fb_series, mae_series = [fb0], [mae0]
    for _ in range(rounds):
        ip = pick_interest_point(focal, model)
        if partner == 'ai':
            reports = ai.report_beliefs(ip, QUERY_RADIUS, focal)
            apply_report(focal, reports, ai_trust, ai.unique_id)
        else:
            reports = mate.report_beliefs(ip, QUERY_RADIUS)
            apply_report(focal, reports, hh_trust, mate.unique_id)
            # Bidirectional exchange: the partner also queries the focal
            # human and updates (the human-human interaction reference)
            ip2 = pick_interest_point(mate, model)
            back = focal.report_beliefs(ip2, QUERY_RADIUS)
            apply_report(mate, back, mate.trust.get(focal.unique_id,
                                                    BASE['initial_trust']),
                         focal.unique_id)
        fb, mae = measure(focal, model, bias_cells)
        fb_series.append(fb)
        mae_series.append(mae)
    return {'false_belief': fb_series, 'mae': mae_series}


def main():
    ap = argparse.ArgumentParser(description='M6 dyadic docking experiment')
    ap.add_argument('--n-seeds', type=int, default=20)
    ap.add_argument('--rounds', type=int, default=40)
    ap.add_argument('--save-dir', default='dyadic_results')
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    results = {}   # (type, condition) -> {'fb': [n_seeds x rounds+1], 'mae': ...}

    conditions = [('ai', a) for a in ALPHAS] + [('human', None)]
    for agent_type in ('exploitative', 'exploratory'):
        for partner, alpha in conditions:
            key = f'{agent_type}|{partner}|{alpha}'
            fb_runs, mae_runs = [], []
            for seed in range(args.n_seeds):
                r = run_dyad(seed, agent_type, partner,
                             alpha if alpha is not None else 0.0, args.rounds)
                fb_runs.append(r['false_belief'])
                mae_runs.append(r['mae'])
            results[key] = {'false_belief': fb_runs, 'mae': mae_runs}
            fb_end = np.mean([r[-1] for r in fb_runs])
            fb_start = np.mean([r[0] for r in fb_runs])
            label = f'alpha={alpha}' if partner == 'ai' else 'human-human'
            print(f'{agent_type:<13} {label:<12} false-belief '
                  f'{fb_start:.2f} -> {fb_end:.2f}   '
                  f'MAE -> {np.nanmean([r[-1] for r in mae_runs]):.2f}')

    json_path = os.path.join(args.save_dir, 'dyadic_docking.json')
    with open(json_path, 'w') as f:
        json.dump({'alphas': ALPHAS, 'rounds': args.rounds,
                   'n_seeds': args.n_seeds, 'base': {k: v for k, v in BASE.items()},
                   'false_epicenter': list(FALSE_EPICENTER),
                   'results': results}, f)
    print(f'Saved {json_path}')

    # ── SI figure: (a) bias trajectories, (b) final bias vs alpha ──────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for row, agent_type in enumerate(('exploitative', 'exploratory')):
        ax = axes[row, 0]
        show = [0.0, 0.5, 1.0]
        colors = plt.cm.viridis(np.linspace(0, 0.85, len(show)))
        for c, a in zip(colors, show):
            runs = np.array(results[f'{agent_type}|ai|{a}']['false_belief'])
            m, s = runs.mean(axis=0), runs.std(axis=0)
            x = np.arange(runs.shape[1])
            ax.plot(x, m, color=c, lw=2, label=f'human×AI, α={a}')
            ax.fill_between(x, m - s, m + s, color=c, alpha=0.15)
        hh = np.array(results[f'{agent_type}|human|None']['false_belief'])
        ax.plot(np.arange(hh.shape[1]), hh.mean(axis=0), 'k--', lw=2,
                label='human×human')
        ax.fill_between(np.arange(hh.shape[1]),
                        hh.mean(axis=0) - hh.std(axis=0),
                        hh.mean(axis=0) + hh.std(axis=0), color='k', alpha=0.1)
        ax.set_xlabel('Interaction round')
        ax.set_ylabel('False-belief strength\n(mean level, truth = 0)')
        ax.set_title(f'{agent_type.capitalize()} focal human — bias trajectory')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[row, 1]
        finals = [np.array(results[f'{agent_type}|ai|{a}']['false_belief'])[:, -1]
                  for a in ALPHAS]
        m = [v.mean() for v in finals]
        s = [v.std(ddof=1) / np.sqrt(len(v)) for v in finals]
        ax.errorbar(ALPHAS, m, yerr=s, fmt='-o', color='#B45F06', capsize=3,
                    label='human×AI (final bias)')
        hh_final = np.array(results[f'{agent_type}|human|None']['false_belief'])[:, -1]
        hm = hh_final.mean()
        hs = hh_final.std(ddof=1) / np.sqrt(len(hh_final))
        ax.axhline(hm, color='k', ls='--', label='human×human (final bias)')
        ax.axhspan(hm - hs, hm + hs, color='k', alpha=0.1)
        start = np.array(results[f'{agent_type}|ai|0.0']['false_belief'])[:, 0].mean()
        ax.axhline(start, color='gray', ls=':', label='implanted bias (round 0)')
        ax.set_xlabel('AI alignment α')
        ax.set_ylabel('Final false-belief strength')
        ax.set_title(f'{agent_type.capitalize()} — final bias vs α')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    for i, ax in enumerate(axes.flat):
        ax.text(-0.12, 1.06, chr(ord('a') + i), transform=ax.transAxes,
                fontsize=15, fontweight='bold', va='bottom', ha='left')
    fig.suptitle('Dyadic docking (M6): human×AI feedback loop vs human×human '
                 'pair\n(no network, no relief; bias implanted at a false '
                 'epicenter, ground truth 0)', fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(args.save_dir, 'dyadic_docking.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {fig_path}')

    # ── Docking verdict ────────────────────────────────────────────────────
    print('\nDocking checks (qualitative reproduction of Glickman & Sharot):')
    for agent_type in ('exploitative', 'exploratory'):
        f0 = np.array(results[f'{agent_type}|ai|0.0']['false_belief'])[:, -1].mean()
        f1 = np.array(results[f'{agent_type}|ai|1.0']['false_belief'])[:, -1].mean()
        hh = np.array(results[f'{agent_type}|human|None']['false_belief'])[:, -1].mean()
        start = np.array(results[f'{agent_type}|ai|0.0']['false_belief'])[:, 0].mean()
        c_mono = f1 > f0
        c_conf = f1 > hh
        print(f'  {agent_type}: start {start:.2f} | final: truthful AI {f0:.2f}, '
              f'confirming AI {f1:.2f}, human-human {hh:.2f}')
        print(f'    bias grows with α: {"PASS" if c_mono else "FAIL"};  '
              f'aligned AI retains more bias than human pair: '
              f'{"PASS" if c_conf else "FAIL"}')
        print(f'    (descriptive) truthful-AI correction vs human pair: '
              f'{f0 - hh:+.2f} — a positive value for exploiters reflects the '
              f'D/δ acceptance window rejecting disconfirming reports')


if __name__ == '__main__':
    main()
