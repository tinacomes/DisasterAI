"""
Goldilocks AI Alignment Experiment: Social vs. AI Filter Bubble Interplay

Research Question:
What is the optimal AI alignment level α* that breaks social echo chambers
without creating AI-enforced filter bubbles?

Background — Triple Social Bubbles:
1. Ego-network bubble: agents share/accept info primarily from friends
2. Agent-type cluster: exploiters cluster with exploiters, explorers with explorers
3. Shared-observation bubble: agents near same cells converge via direct sensing

AI Alignment Interplay:
- High alignment (AI confirms user beliefs): AI penetrates networks but amplifies all three bubbles
- Low alignment (AI reports truth): AI may be rejected (D/delta mechanism rejects divergent info)
- Goldilocks α*: AI is trusted and used, but divergent enough to disrupt bubble consensus

Metrics:
- SECI (Social Echo Chamber Index): -1 to +1
  * -1 = Strong echo chamber (friends very similar)
  *  0 = No echo chamber effect
  * +1 = Anti-echo chamber (friends more diverse)
- AECI (AI Echo Chamber Index): -1 to +1  (same variance formula as SECI)
  * -1 = AI-reliant agents have much lower belief variance than global (AI bubble)
  *  0 = No AI echo chamber effect
  * +1 = AI-reliant agents are more belief-diverse than global
- total_bubble = |SECI| + |AECI|  (minimise both)
- Belief MAE: accuracy cost at each alignment level
- Unmet needs: high-need cells (≥L4) that received zero relief tokens per tick
- Targeting precision: fraction of relief sent to genuinely high-need cells

Goldilocks detection: argmin of total_bubble across alignment sweep

Run locally (3 replications, ~15 min):
    python3 test_filter_bubbles.py

For large-N CI runs use simulate.py + plot_results.py instead.
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from DisasterAI_Model import DisasterModel, HumanAgent
import os

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

base_params = {
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
    'rumor_probability': 1.0,
}

ALIGNMENT_SWEEP    = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
STEADY_STATE_WINDOW = 15

N_RUNS        = 10   # replications for primary alignment sweep
N_FACTOR_RUNS = 2    # replications for factor sweeps

FACTOR_ALPHA       = 0.5
RUMOR_SWEEP        = [0.0, 0.5, 1.0]
DISASTER_SWEEP     = [0, 2, 3]
EXPLOITATIVE_SWEEP = [0.2, 0.5, 0.8]

# Gap-scalar sweep for D/δ cognitive polarisation experiment.
# g=0: both agent types identical (cognitive homogeneity null)
# g=1: baseline (D_exploit=2.0, D_explor=4.0, δ_exploit=3.5, δ_explor=1.2)
# Invariant maintained for any g>0: D_exploit < D_explor AND δ_exploit > δ_explor
GAP_SWEEP = [0.0, 0.5, 1.0, 1.5]

_D_MID     = 3.0   # (2.0 + 4.0) / 2
_D_HALF    = 1.0   # (4.0 - 2.0) / 2
_DELTA_MID = 2.35  # (3.5 + 1.2) / 2
_DELTA_HALF = 1.15 # (3.5 - 1.2) / 2


def _gap_d_delta(g):
    """Return (d_exploit, delta_exploit, d_explor, delta_explor) for gap scalar g."""
    return (
        max(_D_MID - g * _D_HALF, 0.1),           # d_exploit  — floor at 0.1
        max(_DELTA_MID + g * _DELTA_HALF, 0.1),    # delta_exploit
        _D_MID + g * _D_HALF,                      # d_explor
        max(_DELTA_MID - g * _DELTA_HALF, 0.1),    # delta_explor — floor at 0.1
    )


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def _first_cross(series, threshold, direction='up'):
    """First index where series crosses threshold; returns len(series) if never."""
    for i, v in enumerate(series):
        if np.isnan(v):
            continue
        if direction == 'up'   and v >= threshold:
            return i
        if direction == 'down' and v <= threshold:
            return i
    return len(series)


def _first_sustained_break(series, form_thresh=-0.1, break_thresh=-0.05, sustain=5):
    """First index where series recovers above break_thresh *and stays there* for
    `sustain` consecutive non-NaN ticks after having formed below form_thresh.
    Returns len(series) if the condition is never met.
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
    return len(series)


def _first_sustained_cross(series, threshold, sustain=5, direction='up'):
    """First index where series crosses threshold and stays there for `sustain` ticks."""
    for i, v in enumerate(series):
        if np.isnan(v):
            continue
        if direction == 'up' and v >= threshold:
            window = [w for w in series[i:i + sustain] if not np.isnan(w)]
            if len(window) == sustain and all(w >= threshold for w in window):
                return i
        elif direction == 'down' and v <= threshold:
            window = [w for w in series[i:i + sustain] if not np.isnan(w)]
            if len(window) == sustain and all(w <= threshold for w in window):
                return i
    return len(series)


def run_one_sim(params):
    """Run a single simulation and return per-tick metrics dict."""
    model = DisasterModel(**params)

    seci_exploit, seci_explor           = [], []
    aeci_exploit, aeci_explor           = [], []
    trust_ai_exploit, trust_fri_exploit = [], []
    trust_ai_explor,  trust_fri_explor  = [], []
    ai_query_ratio_exploit              = []   # per-tick fraction of queries sent to AI
    ai_query_ratio_explor               = []
    aeci_var                            = []   # AECI-Var timeseries (used in lifecycle plot)
    info_div_exploit, info_div_explor   = [], []
    mae_exploit,  mae_explor            = [], []
    prec_exploit, prec_explor           = [], []
    metric_ticks = []
    # Cumulative call counters for computing per-tick deltas
    _prev_ai_ex = _prev_tot_ex = _prev_ai_er = _prev_tot_er = 0
    # Accumulate relief tokens per 5-tick window for precision calculation
    _win_ex_correct = _win_ex_total = _win_er_correct = _win_er_total = 0

    W, H = params['width'], params['height']
    cum_aid_grid      = np.zeros((W, H), dtype=float)
    cum_disaster_grid = np.zeros((W, H), dtype=float)

    for tick in range(params['ticks']):
        model.step()

        # Accumulate spatial aid and disaster grids
        tok = np.zeros((W, H), dtype=float)
        for pos, cnt in model.tokens_this_tick.items():
            tok[pos[0], pos[1]] = cnt.get('exploit', 0) + cnt.get('explor', 0)
        cum_aid_grid      += tok
        cum_disaster_grid += model.disaster_grid.astype(float)

        # --- Per-tick AI query ratio (delta of cumulative accum_calls counters) ---
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

        if model.seci_data:
            s = model.seci_data[-1]
            seci_exploit.append(float(s[1]))
            seci_explor.append(float(s[2]))

        if model.aeci_data:
            a = model.aeci_data[-1]
            aeci_exploit.append(float(a[1]))
            aeci_explor.append(float(a[2]))

        if model.trust_stats:
            ts = model.trust_stats[-1]
            trust_ai_exploit.append(float(ts[1]))
            trust_fri_exploit.append(float(ts[2]))
            trust_ai_explor.append(float(ts[4]))
            trust_fri_explor.append(float(ts[5]))

        if model.aeci_variance_data:
            aeci_var.append(float(model.aeci_variance_data[-1][1]))

        if model.info_diversity_data:
            d = model.info_diversity_data[-1]
            info_div_exploit.append(float(d[1]))
            info_div_explor.append(float(d[2]))

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
            ex_errors, er_errors = [], []
            for agent in model.agent_list:
                if not isinstance(agent, HumanAgent):
                    continue
                # Filter to informed beliefs only (exclude default L0 priors)
                informed = [(c, b) for c, b in agent.beliefs.items()
                            if isinstance(b, dict) and b.get('confidence', 0) > 0.1]
                err = float(np.mean([
                    abs(b.get('level', 0) - model.disaster_grid[c])
                    for c, b in informed
                ])) if informed else float('nan')
                if agent.agent_type == 'exploitative':
                    ex_errors.append(err)
                else:
                    er_errors.append(err)
            mae_exploit.append(float(np.nanmean(ex_errors)) if ex_errors else float('nan'))
            mae_explor.append( float(np.nanmean(er_errors)) if er_errors else float('nan'))
            prec_exploit.append(float(_win_ex_correct / _win_ex_total) if _win_ex_total > 0 else 0.0)
            prec_explor.append( float(_win_er_correct / _win_er_total) if _win_er_total > 0 else 0.0)
            _win_ex_correct = _win_ex_total = _win_er_correct = _win_er_total = 0
            metric_ticks.append(tick)

    n = len(seci_exploit)

    # ------------------------------------------------------------------ #
    # Spatial coverage & network-periphery metrics (computed post-loop)   #
    # ------------------------------------------------------------------ #
    ticks_run = params['ticks']
    avg_disaster = cum_disaster_grid / ticks_run
    avg_aid      = cum_aid_grid      / ticks_run
    # Positive = high need but little aid; negative = more aid than disaster level
    coverage_deficit = avg_disaster - avg_aid

    G       = model.social_network
    degrees = dict(G.degree())          # {node_id(int): degree}
    deg_vals = sorted(degrees.values())
    n_agents = len(deg_vals)
    lo_thresh = deg_vals[max(0, n_agents // 4 - 1)]
    hi_thresh = deg_vals[min(n_agents - 1, 3 * n_agents // 4)]

    ex, ey = model.epicenter           # grid coords of disaster centre
    all_dists  = []
    agent_data = []                    # (dist, degree, mae, ai_frac, aid_sent)
    for agent in model.agent_list:
        if not isinstance(agent, HumanAgent):
            continue
        node_id = int(agent.unique_id.split('_')[1])
        deg     = degrees.get(node_id, 0)
        ax, ay  = agent.pos
        dist    = float(np.sqrt((ax - ex) ** 2 + (ay - ey) ** 2))
        all_dists.append(dist)
        err = float(np.mean([
            abs(b.get('level', 0) - model.disaster_grid[c])
            for c, b in agent.beliefs.items()
            if isinstance(b, dict)
        ])) if agent.beliefs else 0.0
        ai_frac  = (agent.accum_calls_ai /
                    max(agent.accum_calls_total, 1))
        aid_sent = float(agent.correct_targets + agent.incorrect_targets)
        agent_data.append((dist, deg, err, ai_frac, aid_sent))

    # Split by spatial distance quartile (Q1 = near, Q4 = far)
    if all_dists:
        dist_q1 = np.percentile(all_dists, 25)
        dist_q3 = np.percentile(all_dists, 75)
    else:
        dist_q1 = dist_q3 = 0.0

    near_mae, near_ai, near_aid = [], [], []
    far_mae,  far_ai,  far_aid  = [], [], []
    lodeg_mae, lodeg_ai, lodeg_aid = [], [], []
    hideg_mae, hideg_ai, hideg_aid = [], [], []

    for dist, deg, err, ai_frac, aid_sent in agent_data:
        if dist <= dist_q1:
            near_mae.append(err); near_ai.append(ai_frac); near_aid.append(aid_sent)
        elif dist >= dist_q3:
            far_mae.append(err);  far_ai.append(ai_frac);  far_aid.append(aid_sent)
        if deg <= lo_thresh:
            lodeg_mae.append(err); lodeg_ai.append(ai_frac); lodeg_aid.append(aid_sent)
        elif deg >= hi_thresh:
            hideg_mae.append(err); hideg_ai.append(ai_frac); hideg_aid.append(aid_sent)

    def _m(lst): return float(np.nanmean(lst)) if lst else float('nan')

    # --- Transition timing: first-crossing scalars ---
    # 1. First tick AI trust *sustains* lead over friend trust for ≥5 consecutive ticks.
    #    Using sustained crossing prevents spurious detections from noisy trust updates.
    trust_cross_exploit = _first_sustained_cross(
        [a - f for a, f in zip(trust_ai_exploit, trust_fri_exploit)], 0.0, sustain=5)
    trust_cross_explor = _first_sustained_cross(
        [a - f for a, f in zip(trust_ai_explor, trust_fri_explor)], 0.0, sustain=5)

    # 2. SECI sustained break: SECI stays above -0.05 for ≥5 ticks after forming < -0.1
    seci_break_exploit = _first_sustained_break(seci_exploit)
    seci_break_explor  = _first_sustained_break(seci_explor)

    # 3. First tick AI query ratio (real calls, not AECI proxy) sustains > 50% for ≥5 ticks
    ai_query50_exploit = _first_sustained_cross(ai_query_ratio_exploit, 0.5, sustain=5)
    ai_query50_explor  = _first_sustained_cross(ai_query_ratio_explor,  0.5, sustain=5)

    return {
        'seci_exploit':            seci_exploit,
        'seci_explor':             seci_explor,
        'aeci_exploit':            aeci_exploit,
        'aeci_explor':             aeci_explor,
        'trust_ai_exploit':        trust_ai_exploit,
        'trust_fri_exploit':       trust_fri_exploit,
        'trust_ai_explor':         trust_ai_explor,
        'trust_fri_explor':        trust_fri_explor,
        'ai_query_ratio_exploit':  ai_query_ratio_exploit,
        'ai_query_ratio_explor':   ai_query_ratio_explor,
        'aeci_var':                aeci_var,
        'info_div_exploit':        info_div_exploit,
        'info_div_explor':         info_div_explor,
        'mae_exploit':             mae_exploit,
        'mae_explor':              mae_explor,
        'prec_exploit':            prec_exploit,
        'prec_explor':             prec_explor,
        'unmet_needs':             [float(v) for v in model.unmet_needs_evolution],
        'metric_ticks':            metric_ticks,
        # Scalar timing values (one per replication, aggregated across reps later)
        'trust_cross_exploit':  float(trust_cross_exploit),
        'trust_cross_explor':   float(trust_cross_explor),
        'seci_break_exploit':   float(seci_break_exploit),
        'seci_break_explor':    float(seci_break_explor),
        'ai_query50_exploit':   float(ai_query50_exploit),
        'ai_query50_explor':    float(ai_query50_explor),
        # Spatial coverage maps (averaged across ticks)
        'coverage_deficit_map': coverage_deficit.tolist(),
        'avg_disaster_map':     avg_disaster.tolist(),
        'avg_aid_map':          avg_aid.tolist(),
        'epicenter':            list(model.epicenter),
        # Network-periphery scalars (spatial distance: near Q1 vs far Q4)
        'near_mae':   _m(near_mae),  'far_mae':   _m(far_mae),
        'near_ai':    _m(near_ai),   'far_ai':    _m(far_ai),
        'near_aid':   _m(near_aid),  'far_aid':   _m(far_aid),
        # Network-periphery scalars (graph degree: low Q1 vs high Q4)
        'lodeg_mae':  _m(lodeg_mae), 'hideg_mae':  _m(hideg_mae),
        'lodeg_ai':   _m(lodeg_ai),  'hideg_ai':   _m(hideg_ai),
        'lodeg_aid':  _m(lodeg_aid), 'hideg_aid':  _m(hideg_aid),
    }


def _aggregate(runs):
    """Compute mean and std across replications for all metrics."""
    ts_keys = [
        'seci_exploit', 'seci_explor', 'aeci_exploit', 'aeci_explor',
        'trust_ai_exploit', 'trust_fri_exploit', 'trust_ai_explor', 'trust_fri_explor',
        'ai_query_ratio_exploit', 'ai_query_ratio_explor',
        'aeci_var',
        'info_div_exploit', 'info_div_explor',
        'mae_exploit', 'mae_explor', 'prec_exploit', 'prec_explor',
        'unmet_needs',
    ]
    scalar_keys = [
        'trust_cross_exploit', 'trust_cross_explor',
        'seci_break_exploit',  'seci_break_explor',
        'ai_query50_exploit',  'ai_query50_explor',
        # Spatial / network-periphery scalars
        'near_mae', 'far_mae', 'near_ai', 'far_ai', 'near_aid', 'far_aid',
        'lodeg_mae', 'hideg_mae', 'lodeg_ai', 'hideg_ai', 'lodeg_aid', 'hideg_aid',
    ]
    result = {
        'metric_ticks': runs[0]['metric_ticks'],
        'n_ticks': len(runs[0]['seci_exploit']),
    }
    for key in ts_keys:
        arrays = []
        for run in runs:
            arrays.append([
                float('nan') if (v is None or (isinstance(v, float) and np.isnan(v))) else v
                for v in run.get(key, [])
            ])
        if not any(arrays):
            result[f'{key}_mean'] = []
            result[f'{key}_std']  = []
            continue
        min_len = min(len(a) for a in arrays)
        mat = np.array([a[:min_len] for a in arrays], dtype=float)
        result[f'{key}_mean'] = np.nanmean(mat, axis=0).tolist()
        result[f'{key}_std']  = np.nanstd( mat, axis=0).tolist()
    n_ticks_val = result['n_ticks']
    for key in scalar_keys:
        vals = np.array([run.get(key, float('nan')) for run in runs], dtype=float)
        result[f'{key}_mean'] = float(np.nanmean(vals))
        result[f'{key}_std']  = float(np.nanstd(vals))
        # Fraction of runs where transition occurred (value strictly < n_ticks)
        valid = vals[~np.isnan(vals)]
        occurred = valid[valid < n_ticks_val] if n_ticks_val > 0 else np.array([])
        result[f'{key}_frac']      = float(len(occurred) / max(len(valid), 1))
        result[f'{key}_cond_mean'] = float(np.nanmean(occurred)) if len(occurred) > 0 else float('nan')
    # Spatial maps: mean across replications (2-D arrays stored as list-of-lists)
    for map_key in ('coverage_deficit_map', 'avg_disaster_map', 'avg_aid_map'):
        maps = [np.array(r[map_key]) for r in runs if map_key in r and r[map_key]]
        if maps:
            result[f'{map_key}_mean'] = np.mean(maps, axis=0).tolist()
    result['epicenter'] = runs[0].get('epicenter', [0, 0])
    return result


def run_replicated(params, n_runs, label=''):
    """Run n_runs independent simulations and return aggregated mean/std dict."""
    print(f"\n{'='*60}")
    print(f"Running: {label}  ({n_runs} replication{'s' if n_runs > 1 else ''})")
    print(f"{'='*60}")
    runs = []
    for i in range(n_runs):
        print(f"  Replicate {i+1}/{n_runs}...")
        runs.append(run_one_sim(params))
    return _aggregate(runs)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def ss(series, window=STEADY_STATE_WINDOW):
    """Steady-state mean: mean of last `window` values, NaN-safe."""
    if not series:
        return float('nan')
    return float(np.nanmean(series[-window:]))


def compute_goldilocks_metrics(all_results):
    """Compute steady-state scalar metrics (mean ± std) from replicated results."""
    metrics = {}
    for alpha, res in zip(ALIGNMENT_SWEEP, all_results):
        def ms(key_e, key_r=None):
            key_r = key_r or key_e.replace('exploit', 'explor')
            m = (ss(res[f'{key_e}_mean']) + ss(res[f'{key_r}_mean'])) / 2
            s = (ss(res[f'{key_e}_std'])  + ss(res[f'{key_r}_std']))  / 2
            return m, s

        seci_m, seci_s = ms('seci_exploit', 'seci_explor')
        aeci_m, aeci_s = ms('aeci_exploit', 'aeci_explor')
        mae_m,  mae_s  = ms('mae_exploit',  'mae_explor')
        prec_m, prec_s = ms('prec_exploit', 'prec_explor')
        unmet_m = ss(res['unmet_needs_mean'])
        unmet_s = ss(res['unmet_needs_std'])

        metrics[alpha] = {
            'seci': seci_m, 'seci_std': seci_s,
            'aeci': aeci_m, 'aeci_std': aeci_s,
            'mae':  mae_m,  'mae_std':  mae_s,
            'prec': prec_m, 'prec_std': prec_s,
            'unmet': unmet_m, 'unmet_std': unmet_s,
            'total_bubble': abs(seci_m) + abs(aeci_m),
        }
    return metrics


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_goldilocks(metrics, all_results, save_dir):
    """2×3 Goldilocks summary with mean ± std error bars."""
    alphas     = ALIGNMENT_SWEEP
    total_vals = [metrics[a]['total_bubble'] for a in alphas]
    best_alpha = alphas[int(np.argmin(total_vals))]
    print(f"\nGoldilocks α* = {best_alpha}  (min total_bubble = {min(total_vals):.3f})")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f'Goldilocks AI Alignment (α*={best_alpha}) — mean ± std, N={N_RUNS} replications\n'
        f'Steady state = mean of last {STEADY_STATE_WINDOW} ticks of each run',
        fontsize=13, fontweight='bold'
    )

    def eb(ax, key, color, ylabel, title, ylim=None):
        means = [metrics[a][key] for a in alphas]
        stds  = [metrics[a][f'{key}_std'] for a in alphas]
        ax.errorbar(alphas, means, yerr=stds, fmt='-o', color=color, linewidth=2,
                    capsize=5, capthick=1.5)
        ax.axvline(best_alpha, color='gold', linestyle='--', linewidth=2, label=f'α*={best_alpha}')
        ax.set_xlabel('AI Alignment Level (α)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    eb(axes[0, 0], 'seci', 'b', 'SECI (-1 to +1)',
       'Social Echo Chamber\n(negative = stronger bubble)', (-1.1, 1.1))
    axes[0, 0].axhline(0, color='k', linestyle=':', alpha=0.5)

    eb(axes[0, 1], 'aeci', 'r', 'AECI (-1 to +1)',
       'AI-Induced Bubble\n(negative = stronger AI bubble)', (-1.1, 1.1))
    axes[0, 1].axhline(0, color='k', linestyle=':', alpha=0.5)

    ax = axes[0, 2]
    ax.plot(alphas, total_vals, 'k-o', linewidth=2.5, label='|SECI|+|AECI|')
    ax.plot(best_alpha, min(total_vals), 'g*', markersize=18, zorder=5, label=f'α*={best_alpha}')
    ax.fill_between(alphas, total_vals, alpha=0.15, color='purple')
    ax.axvline(best_alpha, color='gold', linestyle='--', linewidth=2)
    ax.set_xlabel('α')
    ax.set_ylabel('|SECI| + |AECI|')
    ax.set_title('Total Bubble Intensity\n(minimise to find Goldilocks zone)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    eb(axes[1, 0], 'mae', 'm', 'Mean Absolute Error',
       'Belief Accuracy\n(lower = beliefs closer to ground truth)')

    eb(axes[1, 1], 'unmet', 'darkorange', 'Unmet high-need cells',
       'Unmet Needs (level ≥4, 0 tokens)\n(lower = better disaster response)')

    eb(axes[1, 2], 'prec', 'teal', 'Correct / Total targets',
       'Relief Targeting Precision\n(higher = relief on high-need cells)', (0, 1.05))

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, 'goldilocks_alignment_sweep.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Goldilocks figure saved: {path}")

    _plot_timeseries(all_results, save_dir, best_alpha)
    return best_alpha


def _plot_timeseries(all_results, save_dir, best_alpha=None):
    """3×3 timeseries with ± std shading per alignment level.

    SECI and AECI are shown in separate panels for exploitative and exploratory
    agents so the two agent types can be compared without overlaying lines.
    Layout:
      Row 0: SECI (exploitative) | SECI (exploratory)   | Belief MAE
      Row 1: AECI (exploitative) | AECI (exploratory)   | Unmet needs
      Row 2: Relief precision    | (spare)               | (spare)
    """
    colors = plt.cm.viridis(np.linspace(0, 1, len(ALIGNMENT_SWEEP)))

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle(
        f'Filter Bubble & Delivery Metrics Over Time  (mean ± std, N={N_RUNS})',
        fontsize=13, fontweight='bold'
    )
    ax_seci_ex, ax_seci_er, ax_mae    = axes[0]
    ax_aeci_ex, ax_aeci_er, ax_unmet  = axes[1]
    ax_prec,    ax_sp1,     ax_sp2    = axes[2]
    ax_sp1.axis('off')
    ax_sp2.axis('off')

    for color, (res, alpha) in zip(colors, zip(all_results, ALIGNMENT_SWEEP)):
        tf    = list(range(res['n_ticks']))
        ts    = res['metric_ticks']
        label = f'α={alpha}' + (' ★' if alpha == best_alpha else '')

        # SECI — separate panels per agent type
        for mk, ax_d in [('seci_exploit', ax_seci_ex), ('seci_explor', ax_seci_er)]:
            m = np.array(res[f'{mk}_mean'])
            s = np.array(res[f'{mk}_std'])
            x = tf[:len(m)]
            ax_d.plot(x, m, color=color, linewidth=1.8, label=label)
            ax_d.fill_between(x, m - s, m + s, color=color, alpha=0.2)

        # AECI — separate panels per agent type
        for mk, ax_d in [('aeci_exploit', ax_aeci_ex), ('aeci_explor', ax_aeci_er)]:
            m = np.array(res[f'{mk}_mean'])
            s = np.array(res[f'{mk}_std'])
            x = tf[:len(m)]
            ax_d.plot(x, m, color=color, linewidth=1.8, label=label)
            ax_d.fill_between(x, m - s, m + s, color=color, alpha=0.2)

        # MAE — combined exploit+explor, sampled every 5 ticks
        mae_m = (np.array(res['mae_exploit_mean']) + np.array(res['mae_explor_mean'])) / 2
        mae_s = (np.array(res['mae_exploit_std'])  + np.array(res['mae_explor_std']))  / 2
        x = ts[:len(mae_m)]
        ax_mae.plot(x, mae_m, color=color, linewidth=1.8, label=label)
        ax_mae.fill_between(x, mae_m - mae_s, mae_m + mae_s, color=color, alpha=0.2)

        # Precision — exploit (dashed) and explor (solid), sampled, skip NaN
        for mk, ls, sfx in [('prec_exploit', '--', 'exploit'), ('prec_explor', '-', 'explor')]:
            pm = np.array(res[f'{mk}_mean'])
            ps = np.array(res[f'{mk}_std'])
            valid = ~np.isnan(pm)
            if np.any(valid):
                tv = np.array(ts[:len(pm)])[valid]
                ax_prec.plot(tv, pm[valid], color=color, linewidth=1.5,
                             linestyle=ls, label=f'α={alpha} {sfx}')
                ax_prec.fill_between(tv, (pm - ps)[valid], (pm + ps)[valid],
                                     color=color, alpha=0.15)

        # Unmet needs — per tick
        un_m = np.array(res['unmet_needs_mean'])
        un_s = np.array(res['unmet_needs_std'])
        x = tf[:len(un_m)]
        ax_unmet.plot(x, un_m, color=color, linewidth=1.8, label=label)
        ax_unmet.fill_between(x, un_m - un_s, un_m + un_s, color=color, alpha=0.2)

    for ax, title, ylabel, ylim, hl in [
        (ax_seci_ex, 'SECI Over Time — Exploitative Agents\n(confirmation-biased, narrow acceptance)',
         'SECI (-1 to +1)', (-1.1, 1.1), 0),
        (ax_seci_er, 'SECI Over Time — Exploratory Agents\n(open, wide acceptance)',
         'SECI (-1 to +1)', (-1.1, 1.1), 0),
        (ax_aeci_ex, 'AECI Over Time — Exploitative Agents',
         'AECI (-1 to +1)', (-1.1, 1.1), 0),
        (ax_aeci_er, 'AECI Over Time — Exploratory Agents',
         'AECI (-1 to +1)', (-1.1, 1.1), 0),
        (ax_mae,   'Belief MAE Over Time  (exploit + explor avg)',
         'Mean Absolute Error', (0, None), None),
        (ax_prec,  'Relief Targeting Precision\n(solid=exploratory, dashed=exploitative)',
         'Correct / Total', (0, 1.05), 0.6),
        (ax_unmet, 'Unmet High-Need Cells per Tick (level ≥4, 0 tokens)',
         'Count', (0, None), None),
    ]:
        ax.set_xlabel('Tick')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        lo, hi = ylim
        if lo is not None:
            ax.set_ylim(bottom=lo)
        if hi is not None:
            ax.set_ylim(top=hi)
        if hl is not None:
            ax.axhline(hl, color='k', linestyle=':', alpha=0.4)

    for ax in [ax_seci_ex, ax_seci_er, ax_aeci_ex, ax_aeci_er, ax_mae, ax_unmet]:
        ax.legend(fontsize=8)
    # Precision legend: deduplicate (keep one entry per α)
    hs, ls = ax_prec.get_legend_handles_labels()
    ax_prec.legend(hs[::2], [l.replace(' exploit', '') for l in ls[::2]], fontsize=8)

    plt.tight_layout()
    path = os.path.join(save_dir, 'bubble_timeseries.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Time-series figure saved: {path}")


def plot_factor_comparison(rumor_res, disaster_res, mix_res, save_dir):
    """3×3 bar chart comparing factor effects on bubble & response metrics."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle(
        f'Factor Effects at α={FACTOR_ALPHA}  (mean ± std across {N_FACTOR_RUNS} runs)\n'
        f'SECI & MAE: mean over all ticks;  Unmet needs: cumulative sum over all {base_params["ticks"]} ticks\n'
        'Each column: one factor swept while others held at base values',
        fontsize=12, fontweight='bold'
    )

    factor_cols = [
        ('Rumour probability\n(0=no rumours, 1=all communities)',
         RUMOR_SWEEP, rumor_res),
        ('Disaster tempo\n(0=static, 2=medium, 3=rapid)',
         DISASTER_SWEEP, disaster_res),
        ('Exploitative share\n(fraction of exploitative agents)',
         EXPLOITATIVE_SWEEP, mix_res),
    ]
    row_metrics = [
        ('SECI — mean over all ticks\n(negative = social bubble, 0 = neutral)',
         'seci_exploit', 'mean', (-1.1, 1.1)),
        ('Belief MAE — mean over all ticks\n(lower = beliefs closer to ground truth)',
         'mae_exploit',  'mean', (0, None)),
        (f'Total unmet high-need events — sum over all {base_params["ticks"]} ticks\n'
         '(cells at level ≥4 with zero relief; lower = better response)',
         'unmet_needs',  'sum',  (0, None)),
    ]
    bar_colors = ['#2196F3', '#FF9800', '#4CAF50']

    for col, (factor_label, factor_levels, res_dict) in enumerate(factor_cols):
        for row, (metric_label, metric_key, agg, ylim) in enumerate(row_metrics):
            ax = axes[row, col]
            means, stds = [], []
            for lv in factor_levels:
                res = res_dict[lv]
                if agg == 'sum':
                    m = float(np.nansum(res[f'{metric_key}_mean'])) if res[f'{metric_key}_mean'] else float('nan')
                    s = float(np.nansum(res[f'{metric_key}_std']))  if res[f'{metric_key}_std']  else float('nan')
                else:
                    m = float(np.nanmean(res[f'{metric_key}_mean'])) if res[f'{metric_key}_mean'] else float('nan')
                    s = float(np.nanmean(res[f'{metric_key}_std']))  if res[f'{metric_key}_std']  else float('nan')
                means.append(m)
                stds.append(s if not np.isnan(s) else 0.0)

            x = np.arange(len(factor_levels))
            ax.bar(x, means, yerr=stds,
                   color=bar_colors[:len(factor_levels)],
                   alpha=0.8, capsize=6, edgecolor='white', linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels([str(v) for v in factor_levels], fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')

            if row == 0:
                ax.set_title(factor_label, fontsize=10, fontweight='bold')
            if col == 0:
                ax.set_ylabel(metric_label, fontsize=9)

            lo, hi = ylim
            if lo is not None:
                ax.set_ylim(bottom=lo)
            if hi is not None:
                ax.set_ylim(top=hi)
            if metric_key == 'seci_exploit':
                ax.axhline(0, color='k', linestyle=':', alpha=0.5)

    plt.tight_layout()
    path = os.path.join(save_dir, 'factor_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Factor comparison figure saved: {path}")


# ---------------------------------------------------------------------------
# Transition timing figure
# ---------------------------------------------------------------------------

def plot_transition_timing(all_results, save_dir):
    """2×2 figure: fraction of runs where key behavioral shifts occur, per alignment level.

    Each bar = fraction of replications where the threshold was crossed during the
    simulation.  Bar labels show the mean tick at which crossing happened (only shown
    when the fraction is > 0).  A missing bar means the event never occurred in any run.
    """
    alphas = ALIGNMENT_SWEEP
    x      = np.arange(len(alphas))
    w      = 0.35
    x_str  = [str(a) for a in alphas]

    def _annotate(ax, bars, cond_vals):
        for bar, tick in zip(bars, cond_vals):
            h = bar.get_height()
            if not np.isnan(tick) and h > 0.04:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                        f't={tick:.0f}', ha='center', va='bottom', fontsize=7)

    def _bar(ax, key_e, key_r, c_e, c_r, label_e, label_r, title):
        frac_e = [r.get(f'{key_e}_frac', float('nan')) for r in all_results]
        frac_r = [r.get(f'{key_r}_frac', float('nan')) for r in all_results]
        cond_e = [r.get(f'{key_e}_cond_mean', float('nan')) for r in all_results]
        cond_r = [r.get(f'{key_r}_cond_mean', float('nan')) for r in all_results]
        b_e = ax.bar(x - w / 2, frac_e, w, color=c_e, alpha=0.85, label=label_e)
        b_r = ax.bar(x + w / 2, frac_r, w, color=c_r, alpha=0.65, label=label_r)
        _annotate(ax, b_e, cond_e)
        _annotate(ax, b_r, cond_r)
        ax.set_xlabel('AI Alignment')
        ax.set_ylabel('Fraction of runs\nwhere event occurs')
        ax.set_title(title)
        ax.set_ylim(0, 1.25)
        ax.set_xticks(x)
        ax.set_xticklabels(x_str)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        'Transition Occurrence vs AI Alignment\n'
        '(Bar height = fraction of runs where shift occurs; label = mean tick when it does)',
        fontsize=13, fontweight='bold',
    )

    _bar(axes[0, 0],
         'trust_cross_exploit', 'trust_cross_explor',
         '#8B0000', '#FA8072',
         'Exploitative', 'Exploratory',
         'AI Trust Sustains Lead Over Friend Trust\n(≥5 consecutive ticks)')

    _bar(axes[0, 1],
         'seci_break_exploit', 'seci_break_explor',
         '#1A3A6B', '#6BAED6',
         'Exploitative', 'Exploratory',
         'Social Echo Chamber Breaks (SECI → 0)\n(sustained ≥5 ticks above −0.05)')

    _bar(axes[1, 0],
         'ai_query50_exploit', 'ai_query50_explor',
         '#1B5E20', '#66BB6A',
         'Exploitative', 'Exploratory',
         'AI Queries > 50% of Total Queries\n(real call counts, sustained ≥5 ticks)')

    # Bottom-right: steady-state AI query ratio by alignment level
    ax = axes[1, 1]
    ss_ex  = [ss(r.get('ai_query_ratio_exploit_mean', [])) for r in all_results]
    ss_er  = [ss(r.get('ai_query_ratio_explor_mean',  [])) for r in all_results]
    std_ex = [ss(r.get('ai_query_ratio_exploit_std',  [])) for r in all_results]
    std_er = [ss(r.get('ai_query_ratio_explor_std',   [])) for r in all_results]
    ax.errorbar(alphas, ss_ex, yerr=std_ex, fmt='-o', color='#8B0000',
                linewidth=2, capsize=5, label='Exploitative')
    ax.errorbar(alphas, ss_er, yerr=std_er, fmt='-s', color='#FA8072',
                linewidth=2, capsize=5, label='Exploratory')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.6, label='50% threshold')
    ax.set_xlabel('AI Alignment')
    ax.set_ylabel('AI query share\n(mean ± std, last 15 ticks)')
    ax.set_title(f'Steady-State AI Query Ratio\n(last {STEADY_STATE_WINDOW} ticks avg)')
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(x_str)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, 'transition_timing.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Transition timing figure saved: {path}")


# ---------------------------------------------------------------------------
# AI query preference evolution figure
# ---------------------------------------------------------------------------

def plot_aeci_evolution(all_results, save_dir):
    """1×2 figure: AI query ratio (AECI) timeseries per alignment level.

    Y-axis is fraction of queries directed to AI (0 = all to friends, 1 = all to AI).
    A 0.5 dashed threshold marks when AI queries dominate.
    """
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(ALIGNMENT_SWEEP)))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.suptitle(
        'AI Query Preference Evolution\n(Agent shift from friends to AI)',
        fontsize=13, fontweight='bold',
    )

    for ax, key, title in [
        (axes[0], 'aeci_exploit', 'Exploitative Agents'),
        (axes[1], 'aeci_explor',  'Exploratory Agents'),
    ]:
        for color, (res, alpha) in zip(colors, zip(all_results, ALIGNMENT_SWEEP)):
            mean = np.array(res[f'{key}_mean'])
            std  = np.array(res[f'{key}_std'])
            ticks = np.arange(len(mean))
            ax.plot(ticks, mean, color=color, linewidth=2, label=f'AI Alignment={alpha}')
            ax.fill_between(ticks, mean - std, mean + std, color=color, alpha=0.18)
        ax.axhline(0.5, color='red', linestyle='--', linewidth=1.5, label='50% threshold')
        ax.set_xlabel('Simulation Tick')
        ax.set_ylabel('AECI (AI Query Ratio)')
        ax.set_title(title)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, 'aeci_evolution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"AECI evolution figure saved: {path}")


# ---------------------------------------------------------------------------
# Echo chamber lifecycle figure
# ---------------------------------------------------------------------------

def plot_echo_chamber_lifecycle(all_results, save_dir):
    """3×2 figure: SECI & AECI-Var timeseries (left) + lifecycle bar charts (right).

    Bar charts use first-crossing / peak-based scalars rather than raw tick counts,
    avoiding the "final counts" problem where bars merely reflect run length.
    Duration = fraction of ticks spent below threshold (scale-invariant).
    """
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(ALIGNMENT_SWEEP)))
    alphas = ALIGNMENT_SWEEP

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        'Echo Chamber Lifecycle: Rise and Fall\n'
        '(How filter bubbles form, peak, and dissolve)',
        fontsize=13, fontweight='bold',
    )

    # Left column: timeseries
    ax_seci_exp  = fig.add_subplot(3, 2, 1)
    ax_seci_expl = fig.add_subplot(3, 2, 3)
    ax_aeci_var  = fig.add_subplot(3, 2, 5)
    # Right column: bar charts
    ax_peak      = fig.add_subplot(3, 2, 2)
    ax_when      = fig.add_subplot(3, 2, 4)
    ax_dur       = fig.add_subplot(3, 2, 6)

    CHAMBER_THRESH = -0.1

    peak_exp,  peak_expl  = [], []
    when_exp,  when_expl  = [], []
    dur_exp,   dur_expl   = [], []

    for color, (res, alpha) in zip(colors, zip(all_results, ALIGNMENT_SWEEP)):
        label = f'AI Alignment={alpha}'
        ticks_arr = np.arange(res['n_ticks'])

        for ax, key, title in [
            (ax_seci_exp,  'seci_exploit', 'Exploitative Agents: Echo Chamber Formation & Dissolution'),
            (ax_seci_expl, 'seci_explor',  'Exploratory Agents: Echo Chamber Formation & Dissolution'),
        ]:
            mean = np.array(res[f'{key}_mean'])
            std  = np.array(res[f'{key}_std'])
            t    = ticks_arr[:len(mean)]
            ax.plot(t, mean, color=color, linewidth=1.8, label=label)
            ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.15)

        # AECI-Var timeseries
        av_mean = np.array(res['aeci_var_mean'])
        av_std  = np.array(res['aeci_var_std'])
        t_av    = ticks_arr[:len(av_mean)]
        ax_aeci_var.plot(t_av, av_mean, color=color, linewidth=1.8)
        ax_aeci_var.fill_between(t_av, av_mean - av_std, av_mean + av_std,
                                 color=color, alpha=0.15)

        # Lifecycle scalars from mean series (robust to N=1 replications)
        se_mean = np.array(res['seci_exploit_mean'])
        sr_mean = np.array(res['seci_explor_mean'])
        n = res['n_ticks']

        # Peak = max |SECI|
        peak_exp.append(float(np.nanmax(np.abs(se_mean))) if len(se_mean) else 0.0)
        peak_expl.append(float(np.nanmax(np.abs(sr_mean))) if len(sr_mean) else 0.0)

        # When recovery: first tick the mean SECI rises back above CHAMBER_THRESH
        # after forming below it.  This varies by α (high alignment → slower recovery).
        # The formation peak is always ~tick 4 (initial belief formation), so we show
        # recovery instead — a meaningful discriminator across alignment levels.
        when_exp.append(_first_sustained_break(list(se_mean), sustain=3))
        when_expl.append(_first_sustained_break(list(sr_mean), sustain=3))

        # Duration = fraction of ticks with SECI < CHAMBER_THRESH (scale-invariant)
        dur_exp.append(float(np.mean(se_mean < CHAMBER_THRESH)) if len(se_mean) else 0.0)
        dur_expl.append(float(np.mean(sr_mean < CHAMBER_THRESH)) if len(sr_mean) else 0.0)

    # Timeseries decorations
    for ax, title in [
        (ax_seci_exp,  'Exploitative Agents: Echo Chamber Formation & Dissolution'),
        (ax_seci_expl, 'Exploratory Agents: Echo Chamber Formation & Dissolution'),
    ]:
        ax.axhline(0,            color='gray',  linestyle='--', linewidth=1,   label='Neutral (SECI=0)')
        ax.axhline(CHAMBER_THRESH, color='salmon', linestyle=':',  linewidth=1.2, label='Chamber threshold')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Simulation Tick')
        ax.set_ylabel('SECI (Social Echo Chamber Index)')
        ax.set_ylim(-0.55, 0.25)
        ax.grid(True, alpha=0.3)
    ax_seci_exp.legend(fontsize=7, loc='lower right')
    ax_seci_expl.legend(fontsize=7, loc='lower right')

    ax_aeci_var.axhline(0, color='gray', linestyle='--', linewidth=1, label='Neutral')
    ax_aeci_var.set_title('AI Belief Variance Reduction Over Time', fontsize=10)
    ax_aeci_var.set_xlabel('Simulation Tick')
    ax_aeci_var.set_ylabel('AECI-Var (AI Echo Chamber Index)')
    ax_aeci_var.legend(
        [plt.Line2D([0], [0], color=c, linewidth=2) for c in colors],
        [f'AI Alignment={a}' for a in alphas],
        fontsize=7, loc='lower right',
    )
    ax_aeci_var.grid(True, alpha=0.3)

    # Bar charts
    x      = np.arange(len(alphas))
    w      = 0.38
    x_str  = [str(a) for a in alphas]

    ax_peak.bar(x - w/2, peak_exp,  w, label='Exploitative', color='#8B2020', alpha=0.85)
    ax_peak.bar(x + w/2, peak_expl, w, label='Exploratory',  color='#FA8072', alpha=0.85)
    ax_peak.set_title('Maximum Chamber Strength', fontsize=10)
    ax_peak.set_ylabel('Peak Echo Chamber Strength |SECI|')
    ax_peak.set_xticks(x); ax_peak.set_xticklabels(x_str)
    ax_peak.set_xlabel('AI Alignment')
    ax_peak.legend(fontsize=9); ax_peak.grid(True, alpha=0.3, axis='y')

    n_ticks_val = all_results[0]['n_ticks']
    # Cap at n_ticks so "never recovered" bars still render clearly
    when_exp_plot  = [min(v, n_ticks_val) for v in when_exp]
    when_expl_plot = [min(v, n_ticks_val) for v in when_expl]
    ax_when.bar(x - w/2, when_exp_plot,  w, label='Exploitative', color='#1A3A6B', alpha=0.85)
    ax_when.bar(x + w/2, when_expl_plot, w, label='Exploratory',  color='#6BAED6', alpha=0.85)
    # Annotate bars that never recovered with a "✗"
    for xi, (ve, vr) in enumerate(zip(when_exp, when_expl)):
        if ve >= n_ticks_val:
            ax_when.text(xi - w/2, min(ve, n_ticks_val) + 1, '✗', ha='center', fontsize=9)
        if vr >= n_ticks_val:
            ax_when.text(xi + w/2, min(vr, n_ticks_val) + 1, '✗', ha='center', fontsize=9)
    ax_when.set_title('When Does Echo Chamber Recover?\n(first tick SECI sustains > −0.1)', fontsize=10)
    ax_when.set_ylabel('Tick of first recovery (✗ = never)')
    ax_when.set_ylim(0, n_ticks_val * 1.15)
    ax_when.set_xticks(x); ax_when.set_xticklabels(x_str)
    ax_when.set_xlabel('AI Alignment')
    ax_when.legend(fontsize=9); ax_when.grid(True, alpha=0.3, axis='y')

    ax_dur.bar(x - w/2, dur_exp,  w, label='Exploitative', color='#1B5E20', alpha=0.85)
    ax_dur.bar(x + w/2, dur_expl, w, label='Exploratory',  color='#66BB6A', alpha=0.85)
    ax_dur.set_title('How Long Do Chambers Persist?\n(fraction of ticks with SECI<−0.1)', fontsize=10)
    ax_dur.set_ylabel('Fraction of ticks in chamber')
    ax_dur.set_xticks(x); ax_dur.set_xticklabels(x_str)
    ax_dur.set_xlabel('AI Alignment')
    ax_dur.set_ylim(0, 1.05)
    ax_dur.legend(fontsize=9); ax_dur.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, 'echo_chamber_lifecycle.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Echo chamber lifecycle figure saved: {path}")


# ---------------------------------------------------------------------------
# Spatial coverage & periphery-gap plots
# ---------------------------------------------------------------------------

def plot_spatial_coverage(all_results, metrics, save_dir):
    """
    2-row × 3-col figure comparing spatial coverage across three α levels.

    Row 1 — Coverage Deficit (avg_disaster − avg_aid): red = chronically
             under-served; blue = over-served relative to local severity.
    Row 2 — Average Aid Density: how many tokens per tick reached each cell.

    The epicenter is marked with a white star on every panel.
    """
    alphas = ALIGNMENT_SWEEP
    total_vals = [metrics[a]['total_bubble'] for a in alphas]
    best_alpha  = alphas[int(np.argmin(total_vals))]

    # Choose three representative α values
    show_alphas = [0.0, best_alpha, 1.0]
    # De-duplicate while preserving order
    seen = set()
    show_alphas = [a for a in show_alphas if not (a in seen or seen.add(a))]
    ncols = len(show_alphas)

    fig, axes = plt.subplots(2, ncols, figsize=(5 * ncols, 9))
    fig.suptitle(
        'Spatial Coverage Analysis  (avg over full run)\n'
        'Top: coverage deficit (disaster − aid)  |  Bottom: aid density',
        fontsize=12, fontweight='bold'
    )

    # Gather colour-scale limits across all shown results for consistency
    all_deficits = []
    all_aids     = []
    result_map   = {}
    for alpha, res in zip(alphas, all_results):
        if alpha not in show_alphas:
            continue
        if 'coverage_deficit_map_mean' in res:
            all_deficits.append(np.array(res['coverage_deficit_map_mean']))
            all_aids.append(    np.array(res['avg_aid_map_mean']))
        result_map[alpha] = res

    vdef = max(abs(np.nanmax(all_deficits)), abs(np.nanmin(all_deficits))) if all_deficits else 1.0
    vaid = np.nanmax(all_aids) if all_aids else 1.0

    for col, alpha in enumerate(show_alphas):
        res = result_map.get(alpha)
        ex, ey = res['epicenter'] if res else [0, 0]

        ax_def = axes[0, col] if ncols > 1 else axes[0]
        ax_aid = axes[1, col] if ncols > 1 else axes[1]

        label = f'α={alpha:.1f}'
        if alpha == best_alpha:
            label += '  ← α*'

        if res and 'coverage_deficit_map_mean' in res:
            deficit = np.array(res['coverage_deficit_map_mean']).T  # transpose: x→col, y→row
            aid_map = np.array(res['avg_aid_map_mean']).T

            im1 = ax_def.imshow(deficit, origin='lower', cmap='RdBu_r',
                                vmin=-vdef, vmax=vdef, aspect='auto')
            ax_def.plot(ex, ey, '*', color='white', markersize=14,
                        markeredgecolor='black', markeredgewidth=0.8,
                        label='Epicentre')
            plt.colorbar(im1, ax=ax_def, fraction=0.046, pad=0.04,
                         label='Deficit (disaster − aid)')

            im2 = ax_aid.imshow(aid_map, origin='lower', cmap='Blues',
                                vmin=0, vmax=vaid, aspect='auto')
            ax_aid.plot(ex, ey, '*', color='gold', markersize=14,
                        markeredgecolor='black', markeredgewidth=0.8)
            plt.colorbar(im2, ax=ax_aid, fraction=0.046, pad=0.04,
                         label='Avg tokens / tick')
        else:
            ax_def.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax_def.transAxes)
            ax_aid.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax_aid.transAxes)

        ax_def.set_title(label, fontsize=11)
        ax_def.set_xlabel('Grid x'); ax_def.set_ylabel('Grid y')
        ax_aid.set_xlabel('Grid x'); ax_aid.set_ylabel('Grid y')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, 'spatial_coverage.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Spatial coverage figure saved: {path}")


def plot_periphery_gap(all_results, metrics, save_dir):
    """
    2-row × 2-col figure showing how the periphery gap varies with α.

    Left column  — spatial periphery (far Q4 vs near Q1 agents by distance
                   to the disaster epicentre).
    Right column — network periphery (low-degree Q1 vs high-degree Q4 agents
                   in the Watts–Strogatz social network).

    Row 1 — Belief MAE: are periphery agents less accurate?
    Row 2 — AI-query fraction: do periphery agents compensate via AI?

    A gap that shrinks toward α* is evidence that Goldilocks alignment
    reduces structural inequality in situational awareness.
    """
    alphas = ALIGNMENT_SWEEP
    total_vals  = [metrics[a]['total_bubble'] for a in alphas]
    best_alpha  = alphas[int(np.argmin(total_vals))]

    def _get(res, key):
        return float(res.get(f'{key}_mean', float('nan')))

    near_mae  = [_get(r, 'near_mae')  for r in all_results]
    far_mae   = [_get(r, 'far_mae')   for r in all_results]
    near_ai   = [_get(r, 'near_ai')   for r in all_results]
    far_ai    = [_get(r, 'far_ai')    for r in all_results]

    lodeg_mae = [_get(r, 'lodeg_mae') for r in all_results]
    hideg_mae = [_get(r, 'hideg_mae') for r in all_results]
    lodeg_ai  = [_get(r, 'lodeg_ai')  for r in all_results]
    hideg_ai  = [_get(r, 'hideg_ai')  for r in all_results]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        'Periphery Gap across AI Alignment (α)\n'
        'Left: spatial (far vs near epicentre)  |  Right: network degree (low vs high)',
        fontsize=12, fontweight='bold'
    )

    def _panel(ax, core_vals, periph_vals, core_label, periph_label,
                ylabel, title, ylim=None):
        ax.plot(alphas, core_vals,   'o-', color='steelblue',  linewidth=2,
                label=core_label,   markersize=7)
        ax.plot(alphas, periph_vals, 's--', color='firebrick', linewidth=2,
                label=periph_label, markersize=7)
        # Shade the gap
        ax.fill_between(alphas, core_vals, periph_vals, alpha=0.12, color='orange',
                         label='Gap')
        ax.axvline(best_alpha, color='gold', linestyle='--', linewidth=2,
                   label=f'α*={best_alpha}')
        ax.set_xlabel('AI Alignment Level (α)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    _panel(axes[0, 0], near_mae, far_mae,
           'Near epicentre (Q1)', 'Far epicentre (Q4)',
           'Belief MAE', 'Spatial periphery — Belief Accuracy\n(lower = better)')

    _panel(axes[0, 1], hideg_mae, lodeg_mae,
           'High degree (Q4)', 'Low degree (Q1)',
           'Belief MAE', 'Network periphery — Belief Accuracy\n(lower = better)')

    _panel(axes[1, 0], near_ai, far_ai,
           'Near epicentre (Q1)', 'Far epicentre (Q4)',
           'AI query fraction', 'Spatial periphery — AI Usage\n(higher = more AI queries)',
           ylim=(0, 1))

    _panel(axes[1, 1], hideg_ai, lodeg_ai,
           'High degree (Q4)', 'Low degree (Q1)',
           'AI query fraction', 'Network periphery — AI Usage\n(higher = more AI queries)',
           ylim=(0, 1))

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, 'periphery_gap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Periphery gap figure saved: {path}")


# ---------------------------------------------------------------------------
# Gap-scalar sweep plot
# ---------------------------------------------------------------------------

def plot_gap_sweep(gap_results, save_dir):
    """2×2 figure: effect of cognitive polarisation (gap scalar g) at fixed α.

    For each g value we run the full alignment sweep and record α* (the
    Goldilocks point) plus the minimum total_bubble achieved.  Two additional
    panels show how steady-state SECI and AECI change with g.
    """
    g_values        = sorted(gap_results.keys())
    best_alphas     = []
    min_bubbles     = []
    min_bubble_stds = []
    seci_at_star    = []
    seci_stds       = []
    aeci_at_star    = []
    aeci_stds       = []

    for g in g_values:
        results_g = gap_results[g]['all_results']
        metrics_g = compute_goldilocks_metrics(results_g)
        total_vals = [metrics_g[a]['total_bubble'] for a in ALIGNMENT_SWEEP]
        idx = int(np.argmin(total_vals))
        a_star = ALIGNMENT_SWEEP[idx]
        best_alphas.append(a_star)
        min_bubbles.append(total_vals[idx])
        min_bubble_stds.append(metrics_g[a_star]['seci_std'] + metrics_g[a_star]['aeci_std'])
        seci_at_star.append(metrics_g[a_star]['seci'])
        seci_stds.append(metrics_g[a_star]['seci_std'])
        aeci_at_star.append(metrics_g[a_star]['aeci'])
        aeci_stds.append(metrics_g[a_star]['aeci_std'])

    g_labels = [f'g={g}\n(D_ex={_D_MID-g*_D_HALF:.1f}, δ_ex={_DELTA_MID+g*_DELTA_HALF:.2f})'
                for g in g_values]
    x = np.arange(len(g_values))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        'Cognitive Polarisation Sweep (gap scalar g)\n'
        'g=0: no heterogeneity  |  g=1: baseline  |  g=1.5: strongly polarised',
        fontsize=13, fontweight='bold',
    )

    # Panel 1: α* vs g
    ax = axes[0, 0]
    ax.bar(x, best_alphas, color='steelblue', alpha=0.85, edgecolor='white')
    ax.set_xticks(x); ax.set_xticklabels(g_labels, fontsize=8)
    ax.set_ylabel('Goldilocks α*')
    ax.set_title('Goldilocks Point α* vs Cognitive Polarisation')
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.6, label='α=0.5')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')

    # Panel 2: min total_bubble vs g
    ax = axes[0, 1]
    ax.bar(x, min_bubbles, yerr=min_bubble_stds, color='purple', alpha=0.75,
           edgecolor='white', capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    ax.set_xticks(x); ax.set_xticklabels(g_labels, fontsize=8)
    ax.set_ylabel('min(|SECI| + |AECI|)')
    ax.set_title('Minimum Total Bubble at α*\n(lower = Goldilocks zone is more effective)')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 3: SECI at α* vs g
    ax = axes[1, 0]
    seci_arr = np.array(seci_at_star)
    seci_std_arr = np.array(seci_stds)
    ax.plot(g_values, seci_arr, 'b-o', linewidth=2, markersize=8)
    ax.fill_between(g_values, seci_arr - seci_std_arr, seci_arr + seci_std_arr,
                    color='blue', alpha=0.15, label='±1 SD')
    ax.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel('Gap scalar g')
    ax.set_ylabel('SECI at α*')
    ax.set_title('Social Echo Chamber Strength at α*\n(negative = stronger bubble)')
    ax.set_ylim(-1.1, 0.5)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 4: AECI at α* vs g
    ax = axes[1, 1]
    aeci_arr = np.array(aeci_at_star)
    aeci_std_arr = np.array(aeci_stds)
    ax.plot(g_values, aeci_arr, 'r-o', linewidth=2, markersize=8)
    ax.fill_between(g_values, aeci_arr - aeci_std_arr, aeci_arr + aeci_std_arr,
                    color='red', alpha=0.15, label='±1 SD')
    ax.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel('Gap scalar g')
    ax.set_ylabel('AECI at α*')
    ax.set_title('AI-Induced Bubble Strength at α*\n(negative = stronger AI bubble)')
    ax.set_ylim(-1.1, 0.5)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, 'gap_sweep.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Gap sweep figure saved: {path}")


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------

RESULTS_FILE = 'test_results/experiment_results.json'


def _factor_key(v):
    """JSON keys are always strings; restore original numeric type on load."""
    try:
        i = int(v)
        return i if str(i) == str(v) else float(v)
    except (ValueError, TypeError):
        return float(v)


def save_results(all_results, rumor_results, disaster_results, mix_results,
                 gap_results=None, path=RESULTS_FILE):
    """Persist all aggregated results to JSON for later plot-only reruns."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        'alignment_sweep': ALIGNMENT_SWEEP,
        'all_results': all_results,
        'rumor_results':    {str(k): v for k, v in rumor_results.items()},
        'disaster_results': {str(k): v for k, v in disaster_results.items()},
        'mix_results':      {str(k): v for k, v in mix_results.items()},
        'gap_results':      {str(k): v for k, v in (gap_results or {}).items()},
    }
    with open(path, 'w') as f:
        json.dump(data, f)
    print(f"Results saved → {path}")


def load_results(path=RESULTS_FILE):
    """Load previously saved aggregated results; returns the five result dicts."""
    with open(path) as f:
        data = json.load(f)
    all_results      = data['all_results']
    rumor_results    = {_factor_key(k): v for k, v in data['rumor_results'].items()}
    disaster_results = {_factor_key(k): v for k, v in data['disaster_results'].items()}
    mix_results      = {_factor_key(k): v for k, v in data['mix_results'].items()}
    gap_results      = {_factor_key(k): v for k, v in data.get('gap_results', {}).items()}
    return all_results, rumor_results, disaster_results, mix_results, gap_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Goldilocks AI alignment experiment — run simulations and/or plot results.'
    )
    parser.add_argument(
        '--plots-only', action='store_true',
        help=f'Skip simulations and regenerate all plots from {RESULTS_FILE}',
    )
    parser.add_argument(
        '--results-file', default=RESULTS_FILE,
        help='Path to load/save aggregated results JSON (default: %(default)s)',
    )
    parser.add_argument(
        '--save-dir', default='test_results',
        help='Directory for output PNG files (default: %(default)s)',
    )
    parser.add_argument(
        '--skip-gap', action='store_true',
        help='Skip the gap-scalar sweep (4g × 6α × N_FACTOR_RUNS runs); '
             'use when running the gap sweep in a separate parallel CI job.',
    )
    parser.add_argument(
        '--primary-only', action='store_true',
        help='Re-run only the primary alignment sweep (N_RUNS replications); '
             'reuse factor/gap results from --results-file. Useful for bumping '
             'N_RUNS without repeating the factor/gap sweeps.',
    )
    args = parser.parse_args()

    save_dir = args.save_dir

    if args.plots_only:
        print(f"Loading results from {args.results_file} …")
        all_results, rumor_results, disaster_results, mix_results, gap_results = load_results(args.results_file)
        print("Loaded. Regenerating plots …\n")
    elif args.primary_only:
        if os.path.exists(args.results_file):
            print(f"Loading existing factor/gap results from {args.results_file} …")
            _, rumor_results, disaster_results, mix_results, gap_results = load_results(args.results_file)
            print("Loaded. Re-running primary alignment sweep only.\n")
        else:
            print("No existing results file — factor/gap plots will be skipped.\n")
            rumor_results, disaster_results, mix_results, gap_results = {}, {}, {}, {}
        print('=' * 70)
        print('PRIMARY ALIGNMENT SWEEP ONLY')
        print('=' * 70)
        print(f'Sweeping alignment levels: {ALIGNMENT_SWEEP}')
        print(f'Ticks per run: {base_params["ticks"]}')
        print(f'Replications: {N_RUNS}\n')
        all_results = []
        for alpha in ALIGNMENT_SWEEP:
            params = {**base_params, 'ai_alignment_level': alpha}
            all_results.append(run_replicated(params, N_RUNS, f'Alignment α={alpha:.1f}'))
        save_results(all_results, rumor_results, disaster_results, mix_results,
                     gap_results, args.results_file)
    else:
        print('=' * 70)
        print('GOLDILOCKS ALIGNMENT EXPERIMENT: SOCIAL vs. AI FILTER BUBBLE INTERPLAY')
        print('=' * 70)
        print(f'Sweeping alignment levels: {ALIGNMENT_SWEEP}')
        print(f'Ticks per run: {base_params["ticks"]}')
        print(f'Replications (primary sweep): {N_RUNS}')
        print(f'Replications (factor sweeps): {N_FACTOR_RUNS}')
        print(f'Steady-state window: last {STEADY_STATE_WINDOW} ticks\n')

        # 1. Primary alignment sweep
        all_results = []
        for alpha in ALIGNMENT_SWEEP:
            params = {**base_params, 'ai_alignment_level': alpha}
            all_results.append(run_replicated(params, N_RUNS, f'Alignment α={alpha:.1f}'))

        # 2. Factor sweeps (at fixed α = FACTOR_ALPHA)
        print('\n' + '=' * 70)
        print(f'FACTOR SWEEPS  (all at α={FACTOR_ALPHA})')
        print('=' * 70)

        rumor_results = {}
        for rp in RUMOR_SWEEP:
            params = {**base_params, 'ai_alignment_level': FACTOR_ALPHA, 'rumor_probability': rp}
            rumor_results[rp] = run_replicated(params, N_FACTOR_RUNS, f'Rumour p={rp}')

        disaster_results = {}
        for dd in DISASTER_SWEEP:
            params = {**base_params, 'ai_alignment_level': FACTOR_ALPHA, 'disaster_dynamics': dd}
            disaster_results[dd] = run_replicated(params, N_FACTOR_RUNS, f'Disaster dynamics={dd}')

        mix_results = {}
        for se in EXPLOITATIVE_SWEEP:
            params = {**base_params, 'ai_alignment_level': FACTOR_ALPHA, 'share_exploitative': se}
            mix_results[se] = run_replicated(params, N_FACTOR_RUNS, f'Exploitative share={se}')

        # 3. Gap-scalar sweep (skipped when --skip-gap is set so CI can run it
        #    in a parallel job and avoid hitting the per-job timeout).
        if args.skip_gap:
            print('\nSkipping gap-scalar sweep (--skip-gap); run separately.')
            gap_results = {}
        else:
            print('\n' + '=' * 70)
            print('GAP-SCALAR SWEEP  (cognitive polarisation: g ∈ ' + str(GAP_SWEEP) + ')')
            print('=' * 70)
            gap_results = {}
            for g in GAP_SWEEP:
                d_ex, dlt_ex, d_er, dlt_er = _gap_d_delta(g)
                print(f"\n--- g={g}: D_exploit={d_ex:.2f}, δ_exploit={dlt_ex:.2f}, "
                      f"D_explor={d_er:.2f}, δ_explor={dlt_er:.2f} ---")
                g_alpha_results = []
                for alpha in ALIGNMENT_SWEEP:
                    params = {
                        **base_params,
                        'ai_alignment_level': alpha,
                        'd_exploit':    d_ex,
                        'delta_exploit': dlt_ex,
                        'd_explor':     d_er,
                        'delta_explor': dlt_er,
                    }
                    g_alpha_results.append(
                        run_replicated(params, N_FACTOR_RUNS,
                                       f'g={g} α={alpha:.1f}')
                    )
                gap_results[g] = {'all_results': g_alpha_results}

        save_results(all_results, rumor_results, disaster_results, mix_results,
                     gap_results, args.results_file)

    # --- Plotting (shared by both paths) ---
    metrics = compute_goldilocks_metrics(all_results)

    print('\n' + '=' * 70)
    print('STEADY-STATE METRICS SUMMARY')
    print('=' * 70)
    print(f"{'α':>6}  {'SECI':>8}  {'AECI':>8}  {'|S|+|A|':>8}  {'MAE':>7}  {'Unmet':>7}")
    print('-' * 55)
    min_bubble = min(v['total_bubble'] for v in metrics.values())
    for alpha in ALIGNMENT_SWEEP:
        m = metrics[alpha]
        tag = '  ← α*' if abs(m['total_bubble'] - min_bubble) < 1e-9 else ''
        print(f"{alpha:>6.1f}  {m['seci']:>8.3f}  {m['aeci']:>8.3f}  "
              f"{m['total_bubble']:>8.3f}  {m['mae']:>7.3f}  {m['unmet']:>7.1f}{tag}")

    plot_goldilocks(metrics, all_results, save_dir)
    if rumor_results and disaster_results and mix_results:
        plot_factor_comparison(rumor_results, disaster_results, mix_results, save_dir)
    else:
        print("Factor comparison skipped (no factor-sweep data).")
    plot_transition_timing(all_results, save_dir)
    plot_aeci_evolution(all_results, save_dir)
    plot_echo_chamber_lifecycle(all_results, save_dir)
    plot_spatial_coverage(all_results, metrics, save_dir)
    plot_periphery_gap(all_results, metrics, save_dir)
    if gap_results:
        plot_gap_sweep(gap_results, save_dir)

    print('\n' + '=' * 70)
    print('EXPERIMENT COMPLETE')
    print('=' * 70)
    print('\nInterpretation guide:')
    print('  SECI < 0    : social echo chamber (friends converge more than random)')
    print('  AECI < 0    : AI-induced bubble (AI users more homogeneous than global)')
    print('  total_bubble: |SECI| + |AECI| — minimise to find Goldilocks α*')
    print('  unmet_needs : cells at disaster L4+ with zero relief — measures response failure')
    print('  precision   : fraction of relief correctly sent to high-need cells')
    if not args.plots_only:
        print(f'\nResults saved to {args.results_file} — replot anytime with:')
        print(f'  python3 test_filter_bubbles.py --plots-only')
