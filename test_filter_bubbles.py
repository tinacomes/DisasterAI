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
    'ticks': 200,
    'learning_rate': 0.1,
    'epsilon': 0.3,
    'exploit_trust_lr': 0.015,
    'explor_trust_lr': 0.03,
    'rumor_probability': 1.0,
}

ALIGNMENT_SWEEP    = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
STEADY_STATE_WINDOW = 15

N_RUNS        = 10   # replications for primary alignment sweep
N_FACTOR_RUNS = 2    # replications for one-at-a-time factor sweeps (rumour, disaster, mix)
N_GAP_RUNS    = 10   # replications for cognitive gap sweep; full 250-tick runs give stable steady-state, so N=10 suffices
                     # because α* is selected by argmin of a noisy 11-point composite curve;
                     # N=2 gives SE ≈ σ/√2 ≈ 70% of std, making argmin essentially random

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

    Returns:
        -1           if the series never dropped below form_thresh (chamber never formed)
        len(series)  if a chamber formed but never subsequently recovered
        i            the index of the first sustained recovery otherwise
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
            # SECI and AECI at same cadence as MAE — no repeated values, shares metric_ticks axis
            seci_exploit.append(float(model.seci_data[-1][1]) if model.seci_data else float('nan'))
            seci_explor.append( float(model.seci_data[-1][2]) if model.seci_data else float('nan'))
            aeci_exploit.append(float(model.aeci_data[-1][1]) if model.aeci_data else float('nan'))
            aeci_explor.append( float(model.aeci_data[-1][2]) if model.aeci_data else float('nan'))

            ex_errors, er_errors = [], []
            for agent in model.agent_list:
                if not isinstance(agent, HumanAgent):
                    continue
                # MAE over actual disaster cells only (disaster_grid >= 1).
                # Measures whether the agent found the disaster, not confidence
                # on non-disaster cells (which dominated at high α).
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

    # Cell-based near/far scalars — computed with THIS run's epicentre so they
    # aggregate correctly across replications (unlike averaging maps first).
    _H, _W = coverage_deficit.shape
    _Yg, _Xg = np.mgrid[0:_H, 0:_W]
    _cell_dist = np.sqrt((_Xg - ex) ** 2 + (_Yg - ey) ** 2)
    _cd_flat   = _cell_dist.flatten()
    _cq1 = np.percentile(_cd_flat, 25)
    _cq3 = np.percentile(_cd_flat, 75)
    _near_mask = _cell_dist <= _cq1
    _far_mask  = _cell_dist >= _cq3
    near_cell_deficit = float(np.nanmean(coverage_deficit[_near_mask]))
    far_cell_deficit  = float(np.nanmean(coverage_deficit[_far_mask]))
    near_cell_aid     = float(np.nanmean(avg_aid[_near_mask]))
    far_cell_aid      = float(np.nanmean(avg_aid[_far_mask]))
    all_dists  = []
    agent_data = []                    # (dist, degree, mae, ai_frac, aid_sent)
    for agent in model.agent_list:
        if not isinstance(agent, HumanAgent):
            continue
        node_id = int(agent.unique_id.split('_')[1])
        deg     = degrees.get(node_id, 0)
        # Use initial spawn position, not final position — agents move every tick so
        # their end position is arbitrary relative to epicenter.  initial_pos reflects
        # structural periphery (whether an agent's home area is far from the disaster).
        ax, ay  = getattr(agent, 'initial_pos', agent.pos)
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
        # Cell-based near/far scalars (correct per-run epicentre)
        'near_cell_deficit': near_cell_deficit,
        'far_cell_deficit':  far_cell_deficit,
        'near_cell_aid':     near_cell_aid,
        'far_cell_aid':      far_cell_aid,
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
        'near_cell_deficit', 'far_cell_deficit', 'near_cell_aid', 'far_cell_aid',
        'near_mae', 'far_mae', 'near_ai', 'far_ai', 'near_aid', 'far_aid',
        'lodeg_mae', 'hideg_mae', 'lodeg_ai', 'hideg_ai', 'lodeg_aid', 'hideg_aid',
    ]
    result = {
        'metric_ticks': runs[0]['metric_ticks'],
        'n_ticks': len(runs[0]['unmet_needs']),  # unmet_needs stays per-tick
        'n_runs': len(runs),
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
        # Per-run steady-state values for boxplot display in goldilocks figure
        ss_window = STEADY_STATE_WINDOW
        result[f'{key}_ss_runs'] = [
            float(np.nanmean(a[-ss_window:])) if a else float('nan')
            for a in arrays
        ]
    n_ticks_val = result['n_ticks']
    for key in scalar_keys:
        vals = np.array([run.get(key, float('nan')) for run in runs], dtype=float)
        result[f'{key}_mean'] = float(np.nanmean(vals))
        result[f'{key}_std']  = float(np.nanstd(vals))
        # Preserve per-run values so callers can draw boxplots when N > 1
        result[f'{key}_runs'] = [float(v) for v in vals]
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
    """Compute late-run scalar metrics (mean ± std + per-run lists) from replicated results.

    Primary metrics window: last STEADY_STATE_WINDOW (15) samples of the 5-tick-cadence
    series (SECI, AECI, MAE, prec) — equivalent to the last 75 ticks of a 200-tick run.
    Per-tick series (unmet_needs) use a matching 75-tick window for consistency.

    Returns per-α dicts with three composite scores:
      total_bubble      = |SECI| + |AECI|                (raw, unweighted)
      total_bubble_norm = |SECI_norm| + |AECI_norm|      (range-normalised; both metrics
                          contribute equally regardless of magnitude difference)
      total_score_norm  = |SECI_norm| + |AECI_norm| + MAE_norm  (operational composite:
                          also penalises poor belief accuracy)
    α* from total_bubble_norm is the recommended primary result.
    """
    metrics = {}
    # SECI/MAE/prec are sampled every 5 ticks → 15 samples = 75 ticks late-run avg.
    # aeci_var and unmet_needs are per-tick → use 75-tick window for consistency.
    TICK_WINDOW = STEADY_STATE_WINDOW * 5   # 75 ticks

    for alpha, res in zip(ALIGNMENT_SWEEP, all_results):
        def ms(key_e, key_r=None):
            key_r = key_r or key_e.replace('exploit', 'explor')
            m = (ss(res[f'{key_e}_mean']) + ss(res[f'{key_r}_mean'])) / 2
            s = (ss(res[f'{key_e}_std'])  + ss(res[f'{key_r}_std']))  / 2
            return m, s

        def runs_pair(key_e, key_r=None):
            """Per-run late-run values averaged over exploit+explor, for boxplots."""
            key_r = key_r or key_e.replace('exploit', 'explor')
            re = res.get(f'{key_e}_ss_runs', [])
            rr = res.get(f'{key_r}_ss_runs', [])
            n = max(len(re), len(rr))
            return [
                float(np.nanmean([
                    re[i] if i < len(re) else float('nan'),
                    rr[i] if i < len(rr) else float('nan'),
                ]))
                for i in range(n)
            ]

        seci_m, seci_s = ms('seci_exploit', 'seci_explor')
        # Use variance-based AECI (aeci_var) — same formula as SECI:
        #   community variance vs global variance, but grouping by AI-reliance
        #   rather than social-network community.  This makes the Goldilocks sum
        #   total_bubble = |SECI_var| + |AECI_var| a symmetric, coherent formula.
        # aeci_var is per-tick → use 75-tick window for consistency with SECI.
        aeci_m = ss(res['aeci_var_mean'], TICK_WINDOW)
        aeci_s = ss(res['aeci_var_std'],  TICK_WINDOW)
        mae_m,  mae_s  = ms('mae_exploit',  'mae_explor')
        prec_m, prec_s = ms('prec_exploit', 'prec_explor')
        # Per-tick series: use 75-tick window to match SECI cadence
        unmet_m = ss(res['unmet_needs_mean'], TICK_WINDOW)
        unmet_s = ss(res['unmet_needs_std'],  TICK_WINDOW)

        metrics[alpha] = {
            'seci': seci_m, 'seci_std': seci_s, 'seci_runs': runs_pair('seci_exploit', 'seci_explor'),
            'aeci': aeci_m, 'aeci_std': aeci_s, 'aeci_runs': res.get('aeci_var_ss_runs', []),
            'mae':  mae_m,  'mae_std':  mae_s,  'mae_runs':  runs_pair('mae_exploit',  'mae_explor'),
            'prec': prec_m, 'prec_std': prec_s, 'prec_runs': runs_pair('prec_exploit', 'prec_explor'),
            'unmet': unmet_m, 'unmet_std': unmet_s,
            'unmet_runs': res.get('unmet_needs_ss_runs', []),
            'total_bubble': abs(seci_m) + abs(aeci_m),
        }

    # ── Range-normalise |SECI|, |AECI|, MAE across the sweep ────────────────
    # This ensures each metric contributes equally to the composite regardless
    # of its absolute magnitude (|SECI| is typically 3–5× larger than |AECI|).
    def _norm(vals):
        lo, hi = min(vals), max(vals)
        span = hi - lo
        if span < 1e-9:
            return [0.5] * len(vals)   # degenerate: all equal → arbitrary midpoint
        return [(v - lo) / span for v in vals]

    seci_abs = [abs(metrics[a]['seci']) for a in ALIGNMENT_SWEEP]
    aeci_abs = [abs(metrics[a]['aeci']) for a in ALIGNMENT_SWEEP]
    mae_vals  = [metrics[a]['mae']       for a in ALIGNMENT_SWEEP]

    seci_n = _norm(seci_abs)
    aeci_n = _norm(aeci_abs)
    mae_n  = _norm(mae_vals)

    for i, alpha in enumerate(ALIGNMENT_SWEEP):
        metrics[alpha]['seci_norm']          = seci_n[i]
        metrics[alpha]['aeci_norm']          = aeci_n[i]
        metrics[alpha]['mae_norm']           = mae_n[i]
        metrics[alpha]['total_bubble_norm']  = seci_n[i] + aeci_n[i]
        metrics[alpha]['total_score_norm']   = seci_n[i] + aeci_n[i] + mae_n[i]

    return metrics


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_goldilocks(metrics, all_results, save_dir):
    """2×3 Goldilocks summary — boxplots per alpha (N>1) or scatter dots (N=1).
    No lines connect alpha values; each alpha is an independent condition.

    α* is identified from the range-normalised composite so that SECI and AECI
    contribute equally regardless of their absolute magnitude difference.
    The composite panel also shows the operational score (+MAE) so the reader can
    see whether the bubble-minimising α also improves disaster response accuracy.
    """
    alphas = ALIGNMENT_SWEEP

    # Primary objective: range-normalised bubble composite
    norm_vals  = [metrics[a]['total_bubble_norm'] for a in alphas]
    score_vals = [metrics[a]['total_score_norm']  for a in alphas]
    best_alpha       = alphas[int(np.argmin(norm_vals))]
    best_alpha_score = alphas[int(np.argmin(score_vals))]
    print(f"\nGoldilocks α* = {best_alpha}  "
          f"(min normalised bubble = {min(norm_vals):.3f})")
    if best_alpha_score != best_alpha:
        print(f"  Note: operational α* (+MAE) = {best_alpha_score} "
              f"(bubble and response objectives disagree)")
    else:
        print(f"  Operational α* (+MAE) = {best_alpha_score}  ✓ objectives agree")

    use_box = all_results[0].get('n_runs', N_RUNS) > 1
    n_actual = all_results[0].get('n_runs', N_RUNS)
    late_run_label = f'Late-run avg (last 75 ticks, {STEADY_STATE_WINDOW} samples)'

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f'Goldilocks AI Alignment (α*={best_alpha}) — '
        f'{"boxplots" if use_box else "single run"}, N={n_actual} replications\n'
        f'{late_run_label}',
        fontsize=13, fontweight='bold'
    )

    box_w = max(0.03, (max(alphas) - min(alphas)) / len(alphas) * 0.35)

    def eb(ax, key, color, ylabel, title, ylim=None, hline=None):
        means = [metrics[a][key] for a in alphas]
        stds  = [metrics[a][f'{key}_std'] for a in alphas]
        runs  = [metrics[a].get(f'{key}_runs', []) for a in alphas]
        if use_box and any(len(r) > 1 for r in runs):
            bp = ax.boxplot(runs, positions=alphas, widths=box_w,
                            patch_artist=True, showfliers=True, manage_ticks=False)
            for patch in bp['boxes']:
                patch.set_facecolor(color); patch.set_alpha(0.4)
            for el in ('whiskers', 'caps', 'medians', 'fliers'):
                for item in bp[el]:
                    item.set_color(color)
            ax.scatter(alphas, means, marker='x', color=color, s=60, zorder=5,
                       linewidths=2, label='Mean')
        else:
            ax.scatter(alphas, means, color=color, s=80, zorder=5)
            for xi, mi, si in zip(alphas, means, stds):
                if si > 0:
                    ax.errorbar(xi, mi, yerr=si, fmt='none', color=color,
                                capsize=5, capthick=1.5, linewidth=1.5)
        ax.axvline(best_alpha, color='gold', linestyle='--', linewidth=2, label=f'α*={best_alpha}')
        if hline is not None:
            ax.axhline(hline, color='k', linestyle=':', alpha=0.5)
        ax.set_xlabel('AI Alignment Level (α)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    eb(axes[0, 0], 'seci', 'b', 'SECI (-1 to +1)',
       'Social Echo Chamber\n(negative = stronger bubble)', (-1.1, 1.1), hline=0)

    eb(axes[0, 1], 'aeci', 'r', 'AECI-Var (-1 to +1)',
       'AI-Induced Bubble (variance-based)\n(negative = AI users more homogeneous than global)',
       (-1.1, 1.1), hline=0)

    # ── Goldilocks composite panel ──────────────────────────────────────────
    # Shows two normalised objectives so the reader can see whether the bubble-
    # minimising α also minimises belief error.  If both argmins coincide the
    # Goldilocks claim is operationally grounded; if they differ it's a nuance.
    ax = axes[0, 2]
    ax.scatter(alphas, norm_vals,  color='purple',     s=80,  zorder=5,
               label='|SECI|+|AECI| (norm.)')
    ax.plot(alphas,    norm_vals,  color='purple',     lw=1,  alpha=0.35, linestyle='--')
    ax.scatter(alphas, score_vals, color='darkorange',  s=60,  zorder=5, marker='s',
               label='|SECI|+|AECI|+MAE (norm.)')
    ax.plot(alphas,    score_vals, color='darkorange',  lw=1,  alpha=0.35, linestyle=':')
    ax.axvline(best_alpha, color='gold', linestyle='--', linewidth=2,
               label=f'α*(bubble)={best_alpha}')
    if best_alpha_score != best_alpha:
        ax.axvline(best_alpha_score, color='sienna', linestyle=':', linewidth=2,
                   label=f'α*(+MAE)={best_alpha_score}')
    ax.set_xlabel('α')
    ax.set_ylabel('Normalised composite (0 = best, 2 = worst)')
    ax.set_title('Goldilocks Composite (range-normalised)\n'
                 'Purple: bubble-only  |  Orange: + MAE penalty')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    eb(axes[1, 0], 'mae', 'm', 'Mean Absolute Error',
       'Belief Accuracy\n(lower = beliefs closer to ground truth)')

    eb(axes[1, 1], 'unmet', 'darkorange', 'Unmet high-need cells',
       'Unmet Needs (level ≥3, 0 tokens)\n(lower = better disaster response)')

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
        f'Filter Bubble & Delivery Metrics Over Time  '
        f'(mean ± std, N={all_results[0].get("n_runs", N_RUNS)})',
        fontsize=13, fontweight='bold'
    )
    ax_seci_ex, ax_seci_er, ax_mae    = axes[0]
    ax_aeci_ex, ax_aeci_er, ax_unmet  = axes[1]
    ax_prec,    ax_aqr_ex,  ax_aqr_er  = axes[2]

    for color, (res, alpha) in zip(colors, zip(all_results, ALIGNMENT_SWEEP)):
        tf    = list(range(res['n_ticks']))
        ts    = res['metric_ticks']
        label = f'α={alpha}' + (' ★' if alpha == best_alpha else '')

        # SECI — sampled at metric_ticks cadence.
        # Skip index 0 (tick 0 value is always 0 before any community variance
        # is computed) to avoid a misleading visual drop from 0 to first real value.
        for mk, ax_d in [('seci_exploit', ax_seci_ex), ('seci_explor', ax_seci_er)]:
            m = np.array(res[f'{mk}_mean'])
            s = np.array(res[f'{mk}_std'])
            x = np.array(ts[:len(m)])
            i0 = 1 if len(m) > 1 else 0   # start at first real measurement
            ax_d.plot(x[i0:], m[i0:], color=color, linewidth=1.8, label=label)
            ax_d.fill_between(x[i0:], m[i0:] - s[i0:], m[i0:] + s[i0:], color=color, alpha=0.2)

        # AECI — sampled at metric_ticks cadence; same skip for same reason.
        for mk, ax_d in [('aeci_exploit', ax_aeci_ex), ('aeci_explor', ax_aeci_er)]:
            m = np.array(res[f'{mk}_mean'])
            s = np.array(res[f'{mk}_std'])
            x = np.array(ts[:len(m)])
            i0 = 1 if len(m) > 1 else 0
            ax_d.plot(x[i0:], m[i0:], color=color, linewidth=1.8, label=label)
            ax_d.fill_between(x[i0:], m[i0:] - s[i0:], m[i0:] + s[i0:], color=color, alpha=0.2)

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

        # AI query ratio — per tick (key: do explorers switch from AI to social at high α?)
        for mk, ax_d in [('ai_query_ratio_exploit', ax_aqr_ex),
                          ('ai_query_ratio_explor',  ax_aqr_er)]:
            qm = np.array(res.get(f'{mk}_mean', []))
            qs = np.array(res.get(f'{mk}_std',  []))
            if len(qm) == 0:
                continue
            valid = ~np.isnan(qm)
            if np.any(valid):
                tv = np.array(tf[:len(qm)])[valid]
                ax_d.plot(tv, qm[valid], color=color, linewidth=1.5, label=label)
                ax_d.fill_between(tv, (qm - qs)[valid], (qm + qs)[valid],
                                  color=color, alpha=0.15)

    for ax, title, ylabel, ylim, hl in [
        (ax_seci_ex, 'SECI Over Time — Exploitative Agents\n(community variance vs global)',
         'SECI (-1 to +1)', (-1.1, 1.1), 0),
        (ax_seci_er, 'SECI Over Time — Exploratory Agents\n(community variance vs global)',
         'SECI (-1 to +1)', (-1.1, 1.1), 0),
        (ax_aeci_ex, 'AECI Over Time — Exploitative Agents\n(AI-heavy vs AI-light within type)',
         'AECI (-1 to +1)', (-1.1, 1.1), 0),
        (ax_aeci_er, 'AECI Over Time — Exploratory Agents\n(AI-heavy vs AI-light within type)',
         'AECI (-1 to +1)', (-1.1, 1.1), 0),
        (ax_mae,   'Belief MAE Over Time  (exploit + explor avg, informed beliefs only)',
         'Mean Absolute Error', (0, None), None),
        (ax_prec,  'Relief Targeting Precision\n(solid=exploratory, dashed=exploitative)',
         'Correct / Total', (0, 1.05), 0.6),
        (ax_unmet, 'Unmet High-Need Cells per Tick (level ≥3, 0 tokens)',
         'Count', (0, None), None),
        (ax_aqr_ex, 'AI Query Ratio — Exploitative Agents\n(fraction of queries sent to AI)',
         'AI / Total queries', (0, 1.05), None),
        (ax_aqr_er, 'AI Query Ratio — Exploratory Agents\n(↓ at high α = switch to social)',
         'AI / Total queries', (0, 1.05), None),
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

    for ax in [ax_seci_ex, ax_seci_er, ax_aeci_ex, ax_aeci_er, ax_mae,
               ax_unmet, ax_aqr_ex, ax_aqr_er]:
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

    # Echo chamber steady-state depth — replaces the "SECI → 0 recovery" panel.
    # Recovery detection was unreliable: the function returned len(series) (~4) when
    # no recovery occurred, which was always < n_ticks (20), making every run look
    # like it "recovered at t=4".  For explorers SECI never forms a chamber at all;
    # for exploiters at high α the chamber is shallow but does not recover.
    # Steady-state depth directly answers "how strong is the echo chamber?" without
    # requiring an ill-defined recovery event.
    ax01 = axes[0, 1]
    ss_seci_ex  = [abs(ss(r.get('seci_exploit_mean', []))) for r in all_results]
    ss_seci_er  = [abs(ss(r.get('seci_explor_mean',  []))) for r in all_results]
    b_e01 = ax01.bar(x - w / 2, ss_seci_ex, w, color='#1A3A6B', alpha=0.85, label='Exploitative')
    b_r01 = ax01.bar(x + w / 2, ss_seci_er, w, color='#6BAED6', alpha=0.65, label='Exploratory')
    for bar, val in zip(list(b_e01) + list(b_r01), ss_seci_ex + ss_seci_er):
        if val > 0.02:
            ax01.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                      f'{val:.2f}', ha='center', va='bottom', fontsize=7)
    ax01.set_xlabel('AI Alignment')
    ax01.set_ylabel('Mean |SECI| (last 15 ticks)')
    ax01.set_title('Echo Chamber Steady-State Depth\n(higher = stronger bubble; explorers near 0 = no chamber)')
    ax01.set_ylim(0, 1.05)
    ax01.set_xticks(x)
    ax01.set_xticklabels(x_str)
    ax01.legend(fontsize=9)
    ax01.grid(True, alpha=0.3, axis='y')

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
    fig.suptitle('AI Query Preference Evolution', fontsize=13, fontweight='bold')

    for ax, key, title in [
        (axes[0], 'aeci_exploit', 'Exploitative Agents'),
        (axes[1], 'aeci_explor',  'Exploratory Agents'),
    ]:
        for color, (res, alpha) in zip(colors, zip(all_results, ALIGNMENT_SWEEP)):
            mean  = np.array(res[f'{key}_mean'])
            std   = np.array(res[f'{key}_std'])
            ticks = np.array(res['metric_ticks'])
            ax.plot(ticks, mean, color=color, linewidth=2, label=f'AI Alignment={alpha}')
            ax.fill_between(ticks, mean - std, mean + std, color=color, alpha=0.18)
        ax.axhline(0.5, color='red', linestyle='--', linewidth=1.5, label='50% threshold')
        ax.set_xlabel('Simulation Tick')
        ax.set_ylabel('AI Query Ratio (cumulative)')
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
        # SECI is sampled at metric_ticks cadence (every 5 ticks); use the actual
        # tick numbers as the x-axis so the plot spans the full simulation length.
        mt = np.array(res['metric_ticks'])

        for ax, key, title in [
            (ax_seci_exp,  'seci_exploit', 'Exploitative Agents: Echo Chamber Formation & Dissolution'),
            (ax_seci_expl, 'seci_explor',  'Exploratory Agents: Echo Chamber Formation & Dissolution'),
        ]:
            mean = np.array(res[f'{key}_mean'])
            std  = np.array(res[f'{key}_std'])
            t    = mt[:len(mean)]          # actual tick numbers [0, 5, 10, ..., 195]
            ax.plot(t, mean, color=color, linewidth=1.8, label=label)
            ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.15)

        # AECI-Var is recorded every tick — use the full tick range
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

        # Recovery: _first_sustained_break returns an *index* into the series (0..len-1),
        # or len(series) as the "never" sentinel.  Convert to actual tick number.
        def _idx_to_tick(idx, mt_arr, n_t):
            return int(mt_arr[idx]) if idx < len(mt_arr) else n_t

        # _first_sustained_break returns -1 (never formed), len (formed/never recovered), or idx
        raw_exp  = _first_sustained_break(list(se_mean), sustain=3)
        raw_expl = _first_sustained_break(list(sr_mean), sustain=3)
        when_exp.append( -1 if raw_exp  < 0 else _idx_to_tick(raw_exp,  mt, n))
        when_expl.append(-1 if raw_expl < 0 else _idx_to_tick(raw_expl, mt, n))

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
        ax.set_ylim(top=0.25)   # let matplotlib choose the lower limit from data
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
    # -1 = never formed (no bar); n_ticks = formed but never recovered (bar at top + ✗)
    when_exp_plot  = [max(v, 0) for v in when_exp]   # -1 → 0 so bar has zero height
    when_expl_plot = [max(v, 0) for v in when_expl]
    ax_when.bar(x - w/2, [min(v, n_ticks_val) for v in when_exp_plot],
                w, label='Exploitative', color='#1A3A6B', alpha=0.85)
    ax_when.bar(x + w/2, [min(v, n_ticks_val) for v in when_expl_plot],
                w, label='Exploratory',  color='#6BAED6', alpha=0.85)
    # Annotate: ○ = chamber never formed; ✗ = formed but never recovered
    for xi, (ve, vr) in enumerate(zip(when_exp, when_expl)):
        if ve == -1:
            ax_when.text(xi - w/2, 2, '○', ha='center', fontsize=9, color='#1A3A6B')
        elif ve >= n_ticks_val:
            ax_when.text(xi - w/2, n_ticks_val + 1, '✗', ha='center', fontsize=9)
        if vr == -1:
            ax_when.text(xi + w/2, 2, '○', ha='center', fontsize=9, color='#6BAED6')
        elif vr >= n_ticks_val:
            ax_when.text(xi + w/2, n_ticks_val + 1, '✗', ha='center', fontsize=9)
    ax_when.set_title(
        'When Does Echo Chamber Recover?\n'
        '(first tick SECI sustains > −0.1;  ○ = no chamber formed,  ✗ = formed, never recovered)',
        fontsize=9)
    ax_when.set_ylabel('Tick of first recovery')
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
    """
    alphas = ALIGNMENT_SWEEP
    best_bubble = alphas[int(np.argmin([metrics[a]['total_bubble_norm'] for a in alphas]))]
    best_score  = alphas[int(np.argmin([metrics[a]['total_score_norm']  for a in alphas]))]

    # Show baseline + both Goldilocks optima; de-duplicate if they coincide
    show_alphas = [0.0, best_score, best_bubble]
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

        ax_def = axes[0, col] if ncols > 1 else axes[0]
        ax_aid = axes[1, col] if ncols > 1 else axes[1]

        if alpha == 0.0:
            label = f'α={alpha:.1f}  (no alignment)'
        elif alpha == best_score and alpha == best_bubble:
            label = f'α={alpha:.1f}  ← α* (both criteria)'
        elif alpha == best_score:
            label = f'α={alpha:.1f}  ← α*(+MAE)'
        elif alpha == best_bubble:
            label = f'α={alpha:.1f}  ← α*(bubble)'
        else:
            label = f'α={alpha:.1f}'

        if res and 'coverage_deficit_map_mean' in res:
            deficit = np.array(res['coverage_deficit_map_mean']).T  # transpose: x→col, y→row
            aid_map = np.array(res['avg_aid_map_mean']).T

            im1 = ax_def.imshow(deficit, origin='lower', cmap='RdBu_r',
                                vmin=-vdef, vmax=vdef, aspect='auto')
            plt.colorbar(im1, ax=ax_def, fraction=0.046, pad=0.04,
                         label='Deficit (disaster − aid)')

            im2 = ax_aid.imshow(aid_map, origin='lower', cmap='Blues',
                                vmin=0, vmax=vaid, aspect='auto')
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

    Left column  — spatial periphery (cell-based near/far scalars pre-computed
                   per replication in run_one_sim(), then averaged).  Each run
                   uses its own epicentre for the Q1/Q4 distance split, avoiding
                   the blurring artefact of splitting the cross-run mean map.
    Right column — network periphery (low-degree Q1 vs high-degree Q4 agents
                   in the Watts–Strogatz social network).

    Row 1 — Coverage deficit (disaster − aid) near vs far: +ve = under-served.
    Row 2 — Aid density (tokens/tick) near vs far  |  AI-query fraction by degree.
    """
    alphas = ALIGNMENT_SWEEP
    total_n  = [metrics[a]['total_bubble_norm']  for a in alphas]
    total_sn = [metrics[a]['total_score_norm']   for a in alphas]
    best_alpha       = alphas[int(np.argmin(total_n))]   # α*(bubble)
    best_alpha_score = alphas[int(np.argmin(total_sn))]  # α*(bubble + MAE)

    # ── Cell-based spatial metrics — pre-computed per run in run_one_sim() ──
    # Using per-run epicentre avoids the blurring bug: the mean maps average across
    # replications with different epicentre positions, making post-hoc near/far
    # splits against run-0's epicentre meaningless.
    def _get(res, key):
        return float(res.get(f'{key}_mean', float('nan')))

    near_def  = [_get(r, 'near_cell_deficit') for r in all_results]
    far_def   = [_get(r, 'far_cell_deficit')  for r in all_results]
    near_aid_d = [_get(r, 'near_cell_aid')    for r in all_results]
    far_aid_d  = [_get(r, 'far_cell_aid')     for r in all_results]

    lodeg_mae = [_get(r, 'lodeg_mae') for r in all_results]
    hideg_mae = [_get(r, 'hideg_mae') for r in all_results]
    lodeg_aid = [_get(r, 'lodeg_aid') for r in all_results]
    hideg_aid = [_get(r, 'hideg_aid') for r in all_results]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        'Periphery Gap across AI Alignment (α)\n'
        'Left: spatial (cell-based near/far epicentre)  |  Right: network degree (low vs high)',
        fontsize=12, fontweight='bold'
    )

    def _panel(ax, core_vals, periph_vals, core_label, periph_label,
                ylabel, title, ylim=None, hline=None):
        ax.plot(alphas, core_vals,   'o-',  color='steelblue', linewidth=2,
                label=core_label,   markersize=7)
        ax.plot(alphas, periph_vals, 's--', color='firebrick', linewidth=2,
                label=periph_label, markersize=7)
        ax.fill_between(alphas, core_vals, periph_vals, alpha=0.12, color='orange',
                        label='Gap')
        ax.axvline(best_alpha, color='gold', linestyle='--', linewidth=2,
                   label=f'α*(bubble)={best_alpha}')
        ax.axvline(best_alpha_score, color='tomato', linestyle=':', linewidth=1.8,
                   label=f'α*(+MAE)={best_alpha_score}')
        if hline is not None:
            ax.axhline(hline, color='k', linestyle=':', linewidth=1, alpha=0.5)
        ax.set_xlabel('AI Alignment Level (α)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Left column: cell-based spatial periphery
    _panel(axes[0, 0], near_def, far_def,
           'Near epicentre (Q1 cells)', 'Far epicentre (Q4 cells)',
           'Coverage deficit (disaster − aid)',
           'Spatial Coverage Deficit',
           hline=0)

    _panel(axes[1, 0], near_aid_d, far_aid_d,
           'Near epicentre (Q1 cells)', 'Far epicentre (Q4 cells)',
           'Avg tokens / tick',
           'Aid Density near vs far Epicentre')

    # Right column: network-degree periphery
    _panel(axes[0, 1], hideg_mae, lodeg_mae,
           'High degree (Q4)', 'Low degree (Q1)',
           'Belief MAE',
           'Network Periphery — Belief Accuracy')

    _panel(axes[1, 1], hideg_aid, lodeg_aid,
           'High degree (Q4)', 'Low degree (Q1)',
           'Avg aid tokens sent / agent',
           'Network Periphery — Aid Sent per Agent')

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

    g linearly scales the cognitive difference between exploitative and exploratory
    agents from a shared midpoint (D_mid=3.0, δ_mid=2.35):
        d_exploit(g)     = 3.0 − g       δ_exploit(g) = 2.35 + 1.15·g
        d_explor(g)      = 3.0 + g       δ_explor(g)  = 2.35 − 1.15·g
    g=0 → all agents identical at (D=3.0, δ=2.35).
    g=1 (baseline) → (D_ex=2.0, δ_ex=3.5) vs (D_er=4.0, δ_er=1.2).

    α* uses the same range-normalised criteria (total_bubble_norm / total_score_norm)
    as the primary alignment sweep, so g=1.0 reproduces the baseline α*.

    Panel layout:
      (0,0) α* bar chart — directly comparable across g
      (0,1) Raw |SECI|+|AECI| at α* — directly comparable across g
      (1,0) SECI at α* line plot (auto-scaled)
      (1,1) MAE at α* line plot — operational outcome at the Goldilocks point
    """
    g_values          = sorted(gap_results.keys())
    best_alphas       = []   # α*(bubble):  argmin total_bubble_norm
    best_score_alphas = []   # α*(+MAE):    argmin total_score_norm
    seci_at_star      = []
    seci_stds         = []
    aeci_at_star      = []
    aeci_stds         = []
    mae_at_star       = []
    mae_stds          = []

    for g in g_values:
        results_g = gap_results[g]['all_results']
        metrics_g = compute_goldilocks_metrics(results_g)
        # α* uses the same range-normalised criteria as the primary alignment sweep
        # (total_bubble_norm / total_score_norm) so g=1.0 reproduces the baseline α*.
        # N_GAP_RUNS=20 matches N_RUNS, keeping normalisation stable.
        norm_bubble_vals = [metrics_g[a]['total_bubble_norm'] for a in ALIGNMENT_SWEEP]
        idx    = int(np.argmin(norm_bubble_vals))
        a_star = ALIGNMENT_SWEEP[idx]
        best_alphas.append(a_star)
        norm_score_vals  = [metrics_g[a]['total_score_norm']  for a in ALIGNMENT_SWEEP]
        best_score_alphas.append(ALIGNMENT_SWEEP[int(np.argmin(norm_score_vals))])
        seci_at_star.append(metrics_g[a_star]['seci'])
        seci_stds.append(metrics_g[a_star]['seci_std'])
        aeci_at_star.append(metrics_g[a_star]['aeci'])
        aeci_stds.append(metrics_g[a_star]['aeci_std'])
        mae_at_star.append(metrics_g[a_star]['mae'])
        mae_stds.append(metrics_g[a_star]['mae_std'])

    # Raw (un-normalised) bubble at α* — comparable across g
    raw_bubble = [abs(s) + abs(a) for s, a in zip(seci_at_star, aeci_at_star)]

    def _g_label(g):
        d_ex, dlt_ex, d_er, dlt_er = _gap_d_delta(g)
        if g == 0.0:
            return f'g={g:.1f}\n(homogeneous midpoint\nD={d_ex:.1f}, δ={dlt_ex:.2f})'
        if g == 1.0:
            return (f'g={g:.1f}\n(baseline ★\nD_ex={d_ex:.1f}/D_er={d_er:.1f})')
        if g >= 1.4:
            return f'g={g:.1f}\n(strongly polarised)'
        return f'g={g:.1f}'
    g_labels = [_g_label(g) for g in g_values]
    x = np.arange(len(g_values))

    # Suptitle: dynamically show key parameter values
    d0, dlt0, _, _ = _gap_d_delta(0.0)
    d1_ex, dlt1_ex, d1_er, dlt1_er = _gap_d_delta(1.0)
    d15_ex, dlt15_ex, d15_er, dlt15_er = _gap_d_delta(1.5)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        'Cognitive Polarisation Sweep (gap scalar g)\n'
        f'g=0: all agents D={d0:.1f}, δ={dlt0:.2f} (homogeneous midpoint)  |  '
        f'g=1★: D_ex={d1_ex:.1f}/δ_ex={dlt1_ex:.2f} vs D_er={d1_er:.1f}/δ_er={dlt1_er:.2f}  |  '
        f'g=1.5: D_ex={d15_ex:.1f}/δ_ex={dlt15_ex:.2f} vs D_er={d15_er:.1f}/δ_er={dlt15_er:.2f}\n'
        f'(gap jobs: {N_GAP_RUNS} reps × {gap_results[g_values[0]]["all_results"][0].get("n_ticks", "?")} ticks — '
        f'α* selected by range-normalised composite, matching primary sweep criterion)',
        fontsize=10, fontweight='bold',
    )

    # Panel 1: both α* criteria vs g — directly comparable across g
    ax = axes[0, 0]
    w = 0.38
    ax.bar(x - w/2, best_alphas,       w, label='α*(bubble)',
           color='steelblue',  alpha=0.85, edgecolor='white')
    ax.bar(x + w/2, best_score_alphas, w, label='α*(bubble+MAE)',
           color='darkorange', alpha=0.85, edgecolor='white')
    ax.set_xticks(x); ax.set_xticklabels(g_labels, fontsize=8)
    ax.set_ylabel('Goldilocks α*')
    ax.set_title('Goldilocks α* vs Cognitive Polarisation\n'
                 '(blue = bubble-only criterion; orange = bubble + MAE criterion)')
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.6, label='α=0.5')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

    # Panel 2: Raw |SECI|+|AECI| at α* (directly comparable across g)
    ax = axes[0, 1]
    bars = ax.bar(x, raw_bubble, color='darkorchid', alpha=0.80, edgecolor='white')
    # Mark g=1 baseline level
    if 1.0 in g_values:
        baseline_val = raw_bubble[g_values.index(1.0)]
        ax.axhline(baseline_val, color='orange', linestyle='--', linewidth=1.5,
                   label=f'g=1 baseline ({baseline_val:.3f})')
        ax.legend(fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(g_labels, fontsize=8)
    ax.set_ylabel('|SECI| + |AECI|  at α*  (raw, not normalised)')
    ax.set_title('Echo Chamber Strength at Goldilocks Point\n'
                 '(lower = better; directly comparable across g)')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 3: SECI at α* vs g (auto-scaled to show actual variation)
    ax = axes[1, 0]
    seci_arr = np.array(seci_at_star)
    seci_std_arr = np.array(seci_stds)
    ax.plot(g_values, seci_arr, 'b-o', linewidth=2, markersize=8)
    ax.fill_between(g_values, seci_arr - seci_std_arr, seci_arr + seci_std_arr,
                    color='blue', alpha=0.15, label='±1 SD')
    ax.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel('Gap scalar g')
    ax.set_ylabel('SECI at α*')
    ax.set_title('Social Echo Chamber Strength at α*\n(negative = stronger bubble; auto-scaled)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 4: MAE at α* vs g — operational outcome at the Goldilocks point
    ax = axes[1, 1]
    mae_arr = np.array(mae_at_star)
    mae_std_arr = np.array(mae_stds)
    ax.plot(g_values, mae_arr, 'g-o', linewidth=2, markersize=8)
    ax.fill_between(g_values, mae_arr - mae_std_arr, mae_arr + mae_std_arr,
                    color='green', alpha=0.15, label='±1 SD')
    ax.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel('Gap scalar g')
    ax.set_ylabel('Belief MAE at α*')
    ax.set_title('Belief Accuracy at Goldilocks Point\n'
                 '(lower = better; does polarisation hurt accuracy?)')
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
        help='Skip the gap-scalar sweep; use when running it in a separate parallel CI job.',
    )
    parser.add_argument(
        '--gap-only', action='store_true',
        help='Re-run only the gap-scalar sweep (N_GAP_RUNS replications, full ticks) '
             'and regenerate the gap sweep figure.  Loads all other results from '
             '--results-file so the remaining plots are not affected.',
    )
    parser.add_argument(
        '--primary-only', action='store_true',
        help='Re-run only the primary alignment sweep (N_RUNS replications); '
             'reuse factor/gap results from --results-file. Useful for bumping '
             'N_RUNS without repeating the factor/gap sweeps.',
    )
    parser.add_argument(
        '--ticks', type=int, default=None,
        help='Override simulation ticks (default: value in base_params, currently '
             f'{base_params["ticks"]}). Factor/gap sweeps use half this value.',
    )
    parser.add_argument(
        '--n-runs', type=int, default=None,
        help=f'Override primary sweep replications (default: N_RUNS={N_RUNS}).',
    )
    parser.add_argument(
        '--single-alpha', type=float, default=None, metavar='ALPHA',
        help='Run N replications for one alignment level only and save the '
             'aggregated result to --out-file.  Used by the CI matrix job so '
             'each alpha runs in a separate parallel worker.',
    )
    parser.add_argument(
        '--out-file', default=None,
        help='Output JSON path for --single-alpha / --single-gap mode.',
    )
    parser.add_argument(
        '--collect-and-plot', action='store_true',
        help='Load per-alpha JSONs from --results-dir, run factor sweeps, '
             'and produce all plots.  Used after the parallel --single-alpha jobs.',
    )
    parser.add_argument(
        '--collect-gap-and-plot', action='store_true',
        help='Load bubble_gap_*.json files from --results-dir and regenerate '
             'gap_sweep.png only.  Used after parallel --single-gap CI jobs.',
    )
    parser.add_argument(
        '--results-dir', default='filter_bubble_results',
        help='Directory containing per-alpha JSON files for --collect-and-plot.',
    )
    parser.add_argument(
        '--single-factor', nargs=2, default=None, metavar=('TYPE', 'VALUE'),
        help='Run N_FACTOR_RUNS replications for one factor condition and save '
             'the result to --out-file.  TYPE is one of: rumor, disaster, mix. '
             'VALUE is the numeric value.  Used by the CI matrix job.',
    )
    parser.add_argument(
        '--single-gap', type=float, default=None, metavar='G',
        help='Run N_GAP_RUNS replications for one (g, α) pair and save the result '
             'to --out-file.  Requires --gap-alpha.  Used by the CI matrix job.',
    )
    parser.add_argument(
        '--gap-alpha', type=float, default=None, metavar='ALPHA',
        help='Alignment level for --single-gap mode (required when used in CI matrix).',
    )
    args = parser.parse_args()

    # Apply CLI overrides
    if args.ticks is not None:
        base_params['ticks'] = args.ticks
    n_runs_primary = args.n_runs if args.n_runs is not None else N_RUNS
    # Factor/gap sweeps use at most half the primary ticks (they measure relative
    # differences, not absolute steady-state values, so shorter runs suffice)
    factor_ticks = max(50, base_params['ticks'] // 2)

    save_dir = args.save_dir

    # ------------------------------------------------------------------
    # Mode: --single-alpha  (one parallel CI worker per alpha level)
    # ------------------------------------------------------------------
    if args.single_alpha is not None:
        alpha = args.single_alpha
        if alpha not in ALIGNMENT_SWEEP:
            raise ValueError(f'--single-alpha {alpha} not in ALIGNMENT_SWEEP {ALIGNMENT_SWEEP}')
        out = args.out_file or f'filter_bubble_results/bubble_alpha_{alpha}.json'
        os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
        print(f'Single-alpha mode: α={alpha}, ticks={base_params["ticks"]}, n_runs={n_runs_primary}')
        params = {**base_params, 'ai_alignment_level': alpha}
        result = run_replicated(params, n_runs_primary, f'Alignment α={alpha:.1f}')
        with open(out, 'w') as f:
            json.dump({'alpha': alpha, 'result': result}, f)
        print(f'Saved → {out}')
        import sys; sys.exit(0)

    # ------------------------------------------------------------------
    # Mode: --single-factor  (one parallel CI worker per factor condition)
    # ------------------------------------------------------------------
    if args.single_factor is not None:
        ftype, fval_str = args.single_factor
        fval_f = float(fval_str)
        factor_base_sf = {**base_params, 'ticks': factor_ticks}
        if ftype == 'rumor':
            params = {**factor_base_sf, 'ai_alignment_level': FACTOR_ALPHA,
                      'rumor_probability': fval_f}
            label = f'Rumour p={fval_f}'
        elif ftype == 'disaster':
            params = {**factor_base_sf, 'ai_alignment_level': FACTOR_ALPHA,
                      'disaster_dynamics': int(fval_f)}
            label = f'Disaster dynamics={int(fval_f)}'
        elif ftype == 'mix':
            params = {**factor_base_sf, 'ai_alignment_level': FACTOR_ALPHA,
                      'share_exploitative': fval_f}
            label = f'Exploitative share={fval_f}'
        else:
            raise ValueError(f'Unknown factor type {ftype!r}; expected rumor, disaster, or mix')
        print(f'Single-factor mode: type={ftype}, value={fval_f}, '
              f'ticks={factor_ticks}, n_runs={N_FACTOR_RUNS}')
        result = run_replicated(params, N_FACTOR_RUNS, label)
        out = args.out_file or f'filter_bubble_results/bubble_factor_{ftype}_{fval_str}.json'
        os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
        with open(out, 'w') as f:
            json.dump({'factor_type': ftype, 'factor_value': fval_f, 'result': result}, f)
        print(f'Saved → {out}')
        import sys; sys.exit(0)

    # ------------------------------------------------------------------
    # Mode: --single-gap  (one parallel CI worker per g × α cell)
    # ------------------------------------------------------------------
    if args.single_gap is not None:
        g     = args.single_gap
        alpha = args.gap_alpha          # required when used in CI matrix
        d_ex, dlt_ex, d_er, dlt_er = _gap_d_delta(g)
        print(f'Single-gap mode: g={g}, α={alpha}, '
              f'ticks={base_params["ticks"]}, n_runs={N_GAP_RUNS}')
        params = {
            **base_params,
            'ai_alignment_level': alpha,
            'd_exploit':    d_ex,
            'delta_exploit': dlt_ex,
            'd_explor':     d_er,
            'delta_explor': dlt_er,
        }
        result = run_replicated(params, N_GAP_RUNS, f'g={g} α={alpha:.1f}')
        out = args.out_file or \
              f'filter_bubble_results/bubble_gap_{g}_alpha_{alpha:.1f}.json'
        os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
        with open(out, 'w') as f:
            json.dump({'gap': g, 'alpha': alpha, 'result': result}, f)
        print(f'Saved → {out}')
        import sys; sys.exit(0)

    # ------------------------------------------------------------------
    # Mode: --collect-gap-and-plot  (gap sweep CI collect step)
    # ------------------------------------------------------------------
    if args.collect_gap_and_plot:
        import glob as _glob
        results_dir = args.results_dir
        # New format: one file per (g, α) pair — bubble_gap_{g}_alpha_{alpha}.json
        per_cell = {}
        for path in sorted(_glob.glob(os.path.join(results_dir, 'bubble_gap_*_alpha_*.json'))):
            with open(path) as f:
                d = json.load(f)
            per_cell.setdefault(float(d['gap']), {})[float(d['alpha'])] = d['result']
        if not per_cell:
            raise FileNotFoundError(
                f'No bubble_gap_*_alpha_*.json files found in {results_dir}')
        # Assemble into all_results list in ALIGNMENT_SWEEP order for plot_gap_sweep
        gap_results = {}
        for g, alpha_dict in per_cell.items():
            missing = [a for a in ALIGNMENT_SWEEP if a not in alpha_dict]
            if missing:
                print(f'Warning: g={g} missing α levels {missing}')
            gap_results[g] = {
                'all_results': [alpha_dict[a] for a in ALIGNMENT_SWEEP if a in alpha_dict]
            }
        print(f'Loaded gap results for g={sorted(gap_results)}')
        os.makedirs(save_dir, exist_ok=True)
        plot_gap_sweep(gap_results, save_dir)
        print('gap_sweep.png saved.')
        import sys; sys.exit(0)

    # ------------------------------------------------------------------
    # Mode: --collect-and-plot  (runs after all parallel CI jobs)
    # ------------------------------------------------------------------
    if args.collect_and_plot:
        import glob as _glob
        results_dir = args.results_dir
        # Load per-alpha files produced by --single-alpha jobs
        per_alpha = {}
        for path in _glob.glob(os.path.join(results_dir, 'bubble_alpha_*.json')):
            with open(path) as f:
                d = json.load(f)
            per_alpha[float(d['alpha'])] = d['result']
        if not per_alpha:
            raise FileNotFoundError(f'No bubble_alpha_*.json files found in {results_dir}')
        all_results = [per_alpha[a] for a in ALIGNMENT_SWEEP if a in per_alpha]
        missing = [a for a in ALIGNMENT_SWEEP if a not in per_alpha]
        if missing:
            print(f'Warning: missing alpha levels {missing} — plots may be incomplete')
        print(f'Loaded {len(all_results)} alpha results from {results_dir}')

        # Load factor results produced by --single-factor parallel jobs
        print(f'\nLoading factor sweep results from {results_dir} …')
        rumor_results = {}
        disaster_results = {}
        mix_results = {}
        for path in _glob.glob(os.path.join(results_dir, 'bubble_factor_*.json')):
            with open(path) as f:
                d = json.load(f)
            ftype = d['factor_type']
            fval  = d['factor_value']
            if ftype == 'rumor':
                rumor_results[fval] = d['result']
            elif ftype == 'disaster':
                disaster_results[int(fval)] = d['result']
            elif ftype == 'mix':
                mix_results[fval] = d['result']
        print(f'  rumor: {sorted(rumor_results)}, '
              f'disaster: {sorted(disaster_results)}, mix: {sorted(mix_results)}')

        # Load gap results produced by --single-gap parallel jobs (per-(g,α) files)
        per_cell = {}
        for path in _glob.glob(os.path.join(results_dir, 'bubble_gap_*_alpha_*.json')):
            with open(path) as f:
                d = json.load(f)
            per_cell.setdefault(float(d['gap']), {})[float(d['alpha'])] = d['result']
        gap_results = {
            g: {'all_results': [alpha_dict[a] for a in ALIGNMENT_SWEEP if a in alpha_dict]}
            for g, alpha_dict in per_cell.items()
        }
        print(f'  gap scalars: {sorted(gap_results)}')

        save_results(all_results, rumor_results, disaster_results, mix_results,
                     gap_results, os.path.join(results_dir, 'experiment_results.json'))
        # Fall through to plotting below
    elif args.plots_only:
        print(f"Loading results from {args.results_file} …")
        all_results, rumor_results, disaster_results, mix_results, gap_results = load_results(args.results_file)
        print("Loaded. Regenerating plots …\n")
    elif args.gap_only:
        if os.path.exists(args.results_file):
            print(f"Loading existing primary/factor results from {args.results_file} …")
            all_results, rumor_results, disaster_results, mix_results, _ = load_results(args.results_file)
            print("Loaded. Re-running gap sweep only.\n")
        else:
            raise FileNotFoundError(
                f'Results file {args.results_file!r} not found. '
                'Run the full experiment first, then use --gap-only to rerun just the gap sweep.'
            )
        print('=' * 70)
        print(f'GAP-SCALAR SWEEP ONLY  (N_GAP_RUNS={N_GAP_RUNS}, full ticks={base_params["ticks"]})')
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
                    run_replicated(params, N_GAP_RUNS, f'g={g} α={alpha:.1f}')
                )
            gap_results[g] = {'all_results': g_alpha_results}
        save_results(all_results, rumor_results, disaster_results, mix_results,
                     gap_results, args.results_file)
        # Plot only the gap figure and exit
        metrics = compute_goldilocks_metrics(all_results)
        plot_gap_sweep(gap_results, save_dir)
        print('\nGap sweep figure regenerated. Other plots unchanged.')
        import sys; sys.exit(0)
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
        print(f'Replications: {n_runs_primary}\n')
        all_results = []
        for alpha in ALIGNMENT_SWEEP:
            params = {**base_params, 'ai_alignment_level': alpha}
            all_results.append(run_replicated(params, n_runs_primary, f'Alignment α={alpha:.1f}'))
        save_results(all_results, rumor_results, disaster_results, mix_results,
                     gap_results, args.results_file)
    else:
        print('=' * 70)
        print('GOLDILOCKS ALIGNMENT EXPERIMENT: SOCIAL vs. AI FILTER BUBBLE INTERPLAY')
        print('=' * 70)
        print(f'Sweeping alignment levels: {ALIGNMENT_SWEEP}')
        print(f'Ticks per run: {base_params["ticks"]}')
        print(f'Ticks for factor/gap sweeps: {factor_ticks}')
        print(f'Replications (primary sweep): {n_runs_primary}')
        print(f'Replications (factor sweeps): {N_FACTOR_RUNS}')
        print(f'Steady-state window: last {STEADY_STATE_WINDOW} ticks\n')

        # 1. Primary alignment sweep
        all_results = []
        for alpha in ALIGNMENT_SWEEP:
            params = {**base_params, 'ai_alignment_level': alpha}
            all_results.append(run_replicated(params, n_runs_primary, f'Alignment α={alpha:.1f}'))

        # 2. Factor sweeps (at fixed α = FACTOR_ALPHA, shorter ticks — comparative)
        print('\n' + '=' * 70)
        print(f'FACTOR SWEEPS  (all at α={FACTOR_ALPHA}, ticks={factor_ticks})')
        print('=' * 70)
        factor_base = {**base_params, 'ticks': factor_ticks}

        rumor_results = {}
        for rp in RUMOR_SWEEP:
            params = {**factor_base, 'ai_alignment_level': FACTOR_ALPHA, 'rumor_probability': rp}
            rumor_results[rp] = run_replicated(params, N_FACTOR_RUNS, f'Rumour p={rp}')

        disaster_results = {}
        for dd in DISASTER_SWEEP:
            params = {**factor_base, 'ai_alignment_level': FACTOR_ALPHA, 'disaster_dynamics': dd}
            disaster_results[dd] = run_replicated(params, N_FACTOR_RUNS, f'Disaster dynamics={dd}')

        mix_results = {}
        for se in EXPLOITATIVE_SWEEP:
            params = {**factor_base, 'ai_alignment_level': FACTOR_ALPHA, 'share_exploitative': se}
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
            gap_base = {**base_params}   # full ticks — gap sweep needs same warm-up as primary
            for g in GAP_SWEEP:
                d_ex, dlt_ex, d_er, dlt_er = _gap_d_delta(g)
                print(f"\n--- g={g}: D_exploit={d_ex:.2f}, δ_exploit={dlt_ex:.2f}, "
                      f"D_explor={d_er:.2f}, δ_explor={dlt_er:.2f} ---")
                g_alpha_results = []
                for alpha in ALIGNMENT_SWEEP:
                    params = {
                        **gap_base,
                        'ai_alignment_level': alpha,
                        'd_exploit':    d_ex,
                        'delta_exploit': dlt_ex,
                        'd_explor':     d_er,
                        'delta_explor': dlt_er,
                    }
                    g_alpha_results.append(
                        run_replicated(params, N_GAP_RUNS,
                                       f'g={g} α={alpha:.1f}')
                    )
                gap_results[g] = {'all_results': g_alpha_results}

        save_results(all_results, rumor_results, disaster_results, mix_results,
                     gap_results, args.results_file)

    # --- Plotting (shared by both paths) ---
    metrics = compute_goldilocks_metrics(all_results)

    print('\n' + '=' * 70)
    print('LATE-RUN METRICS SUMMARY  (last 75 ticks)')
    print('=' * 70)
    print(f"{'α':>6}  {'SECI':>8}  {'AECI':>8}  {'Bub(norm)':>10}  "
          f"{'Score(norm)':>12}  {'MAE':>7}  {'Unmet':>7}")
    print('-' * 72)
    min_norm  = min(v['total_bubble_norm']  for v in metrics.values())
    min_score = min(v['total_score_norm']   for v in metrics.values())
    for alpha in ALIGNMENT_SWEEP:
        m = metrics[alpha]
        tag = ''
        if abs(m['total_bubble_norm'] - min_norm) < 1e-9:
            tag += '  ← α*(bubble)'
        if abs(m['total_score_norm']  - min_score) < 1e-9:
            tag += '  ← α*(+MAE)' if '← α*(bubble)' not in tag else '+MAE'
        print(f"{alpha:>6.1f}  {m['seci']:>8.3f}  {m['aeci']:>8.3f}  "
              f"{m['total_bubble_norm']:>10.3f}  {m['total_score_norm']:>12.3f}  "
              f"{m['mae']:>7.3f}  {m['unmet']:>7.1f}{tag}")

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
